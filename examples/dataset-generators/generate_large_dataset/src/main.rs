use rand::Rng;
use serde_json::json;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    println!("Generating 1 million vectors with 512 dimensions each...");

    let num_vectors = 1_000_000;
    let dimension = 512;
    let mut rng = rand::thread_rng();

    // Create the JSON structure
    let mut embeddings = Vec::new();

    for i in 0..num_vectors {
        // Generate random vector
        let mut embedding = Vec::new();
        for _ in 0..dimension {
            embedding.push(rng.gen_range(-1.0..1.0));
        }

        embeddings.push(json!({
            "filename": format!("vector_{:07}", i),
            "embedding": embedding
        }));

        // Progress indicator
        if i % 100_000 == 0 {
            println!("Generated {} vectors...", i);
        }
    }

    // Write to file
    let mut file = File::create("large_embeddings.json")?;
    let json_string = serde_json::to_string_pretty(&embeddings)?;
    file.write_all(json_string.as_bytes())?;

    println!("Generated {} vectors in large_embeddings.json", num_vectors);
    println!("File size: {} bytes", json_string.len());
    println!(
        "Estimated disk usage: {:.2} MB",
        json_string.len() as f64 / 1024.0 / 1024.0
    );

    Ok(())
}
