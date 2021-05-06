use prost_build;
use std::path::Path;
use walkdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files: Vec<_> = walkdir::WalkDir::new("../proto")
        .into_iter()
        .filter_map(|dir_entry| {
            dir_entry.ok().and_then(|entry| {
                if entry.file_type().is_dir() {
                    None
                } else {
                    if entry.file_name().to_str().unwrap().ends_with(".proto") {
                        Some(entry.path().to_owned())
                    } else {
                        None
                    }
                }
            })
        })
        .collect();
    prost_build::compile_protos(&files, &[Path::new("../").to_path_buf()])?;

    Ok(())
}
