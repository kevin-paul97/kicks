import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const dataPath = path.join(process.cwd(), "..", "output", "cluster_analysis.json");
    
    if (!fs.existsSync(dataPath)) {
      return NextResponse.json({ error: "Cluster data not found. Run python cluster.py first" }, { status: 404 });
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: "Failed to load cluster data" }, { status: 500 });
  }
}
