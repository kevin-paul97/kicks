"""Fine-tune BigVGAN v2 vocoder on kick drum samples."""

import itertools
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torchaudio
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from torch.utils.data import Dataset, DataLoader

import bigvgan
from bigvgan.env import AttrDict
from bigvgan.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from bigvgan.loss import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    MultiScaleMelSpectrogramLoss,
)
from bigvgan.meldataset import mel_spectrogram

from kicks.model import SAMPLE_RATE, AUDIO_LENGTH

VOCODER_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"
SAVE_DIR = "models/vocoder"


class KickAudioDataset(Dataset):
    """Load raw kick drum waveforms for vocoder fine-tuning."""

    def __init__(self, dir: str) -> None:
        self.waveforms: list[torch.Tensor] = []

        for file in sorted(os.listdir(dir)):
            if not file.endswith(".wav"):
                continue
            audio, sr = torchaudio.load(os.path.join(dir, file))

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)

            # Pad/truncate to fixed length
            if audio.shape[-1] > AUDIO_LENGTH:
                audio = audio[:, :AUDIO_LENGTH]
            elif audio.shape[-1] < AUDIO_LENGTH:
                audio = torch.nn.functional.pad(audio, (0, AUDIO_LENGTH - audio.shape[-1]))

            # Normalize to [-1, 1]
            audio = audio / (audio.abs().max() + 1e-8)
            self.waveforms.append(audio.squeeze(0))  # (AUDIO_LENGTH,)

        print(f"Loaded {len(self.waveforms)} kick samples for vocoder fine-tuning")

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.waveforms[idx]


def finetune(
    data_dir: str = "data/kicks",
    epochs: int = 200,
    batch_size: int = 1,
    lr: float = 1e-4,
    grad_accum: int = 4,
    save_every: int = 50,
) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load pretrained generator
    generator = bigvgan.BigVGAN.from_pretrained(VOCODER_MODEL, use_cuda_kernel=False)
    h = generator.h
    generator = generator.train().to(device)

    # Freeze early layers — only fine-tune last 2 upsampling stages + output conv
    for param in generator.parameters():
        param.requires_grad = False

    num_ups = len(generator.ups)
    unfreeze_from = max(0, num_ups - 2)
    for i in range(unfreeze_from, num_ups):
        for param in generator.ups[i].parameters():
            param.requires_grad = True
        for j in range(generator.num_kernels):
            for param in generator.resblocks[i * generator.num_kernels + j].parameters():
                param.requires_grad = True
    for param in generator.conv_post.parameters():
        param.requires_grad = True
    if hasattr(generator, "activation_post"):
        for param in generator.activation_post.parameters():
            param.requires_grad = True

    # Initialize discriminators from scratch
    mpd = MultiPeriodDiscriminator(h).to(device)
    cqtd = MultiScaleSubbandCQTDiscriminator(h).to(device)

    # Multi-scale mel loss
    mel_loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate)

    # Optimizers (only trainable generator params)
    optim_g = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=lr, betas=(h.adam_b1, h.adam_b2),
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), cqtd.parameters()),
        lr=lr, betas=(h.adam_b1, h.adam_b2),
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    # Dataset
    dataset = KickAudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    gen_total = sum(p.numel() for p in generator.parameters())
    gen_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Generator: {gen_total:,} params ({gen_trainable:,} trainable)")
    print(f"MPD: {sum(p.numel() for p in mpd.parameters()):,} params")
    print(f"CQT-D: {sum(p.numel() for p in cqtd.parameters()):,} params")
    print(f"Training for {epochs} epochs, batch_size={batch_size}, grad_accum={grad_accum}")

    best_mel_loss = float("inf")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Fine-tuning vocoder", total=epochs)

        for epoch in range(1, epochs + 1):
            generator.train()
            mpd.train()
            cqtd.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_mel = 0.0
            n_batches = 0

            for step, wav in enumerate(dataloader):
                wav = wav.to(device).unsqueeze(1)  # (B, 1, T)

                # Compute mel spectrogram from real audio
                mel = mel_spectrogram(
                    wav.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax, center=False,
                ).to(device)

                # ── Discriminator step (generator graph not needed) ──
                with torch.no_grad():
                    wav_gen_d = generator(mel)

                y_df_hat_r, y_df_hat_g, _, _ = mpd(wav, wav_gen_d)
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, _, _ = cqtd(wav, wav_gen_d)
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_d = (loss_disc_f + loss_disc_s) / grad_accum
                loss_d_val = loss_disc_f.item() + loss_disc_s.item()
                loss_d.backward()

                del wav_gen_d, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
                del loss_disc_f, loss_disc_s, loss_d
                if device.type == "mps":
                    torch.mps.empty_cache()

                # ── Generator step (recompute with gradient tracking) ──
                wav_gen = generator(mel)

                # Compute on CPU — BigVGAN mel loss uses float64 mel basis internally,
                # which MPS does not support
                loss_mel = mel_loss_fn(wav.cpu(), wav_gen.cpu()).to(device) * h.lambda_melloss
                mel_val = loss_mel.item()

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav, wav_gen)
                loss_gen_f, _ = generator_loss(y_df_hat_g)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)

                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = cqtd(wav, wav_gen)
                loss_gen_s, _ = generator_loss(y_ds_hat_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

                loss_g = (loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel) / grad_accum
                loss_g_val = loss_gen_f.item() + loss_gen_s.item() + loss_fm_f.item() + loss_fm_s.item() + mel_val
                loss_g.backward()

                del wav_gen, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
                del fmap_f_r, fmap_f_g, fmap_s_r, fmap_s_g
                del loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s, loss_mel, loss_g
                if device.type == "mps":
                    torch.mps.empty_cache()

                # Step optimizers every grad_accum steps
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(mpd.parameters(), cqtd.parameters()),
                        h.clip_grad_norm,
                    )
                    optim_d.step()
                    optim_d.zero_grad()

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in generator.parameters() if p.requires_grad],
                        h.clip_grad_norm,
                    )
                    optim_g.step()
                    optim_g.zero_grad()

                epoch_g_loss += loss_g_val
                epoch_d_loss += loss_d_val
                epoch_mel += mel_val
                n_batches += 1

            scheduler_g.step()
            scheduler_d.step()

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            avg_mel = epoch_mel / max(n_batches, 1)

            progress.update(task, completed=epoch,
                            description=f"G={avg_g:.3f} D={avg_d:.3f} Mel={avg_mel:.3f}")

            # Save best by mel loss
            if avg_mel < best_mel_loss:
                best_mel_loss = avg_mel
                torch.save({
                    "generator": generator.state_dict(),
                    "epoch": epoch,
                    "mel_loss": avg_mel,
                }, os.path.join(SAVE_DIR, "best.pth"))

            # Periodic checkpoint
            if epoch % save_every == 0:
                torch.save({
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "cqtd": cqtd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "epoch": epoch,
                }, os.path.join(SAVE_DIR, f"checkpoint_{epoch}.pth"))
                print(f"\n  Saved checkpoint at epoch {epoch}")

    # Final save
    torch.save({
        "generator": generator.state_dict(),
        "epoch": epochs,
        "mel_loss": best_mel_loss,
    }, os.path.join(SAVE_DIR, "final.pth"))

    print(f"\nDone! Best mel loss: {best_mel_loss:.4f}")
    print(f"Weights saved to {SAVE_DIR}/")


if __name__ == "__main__":
    finetune()
