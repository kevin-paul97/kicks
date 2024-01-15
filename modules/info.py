from torchinfo import summary

def info(model, dset, dloader):
    
    data = iter(dloader)
    batch = next(data)

    print("Max lenght: ", dset.max_length)
    print("Longest sample: ", dset.max_length_path)
    print("Lenght of dataset: ", len(dset.tensors))

    summary(model=model, input_size=(1, 1, dset.max_length))
