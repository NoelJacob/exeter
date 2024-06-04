use std::fs::{create_dir_all, File};
use std::ops::{BitAnd, Rem};
use std::path::PathBuf;

use clap::Parser;
use polars::prelude::*;

/// Program to do interview tasks
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Folder with .ped and .map files
    folder: PathBuf,
}

fn main() {
    let folder = Args::parse().folder;
    let map_fields = vec![
        Field::new("chromosome", DataType::Int32),
        Field::new("snp_id", DataType::String),
        Field::new("genetic_distance", DataType::Int32),
        Field::new("chromosome_position", DataType::Int32),
    ];
    let map_schema = Schema::from_iter(map_fields);

    for f in folder.clone().read_dir().unwrap() {
        let map_file = f.unwrap();
        let fne = map_file.file_name();
        let fname = fne.to_str().unwrap();
        if !fname.ends_with(".map") {
            continue;
        }
        let mut df_map = CsvReadOptions::default()
            .with_has_header(false)
            .with_infer_schema_length(None)
            .with_schema(Some(Arc::new(map_schema.clone())))
            .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
            .try_into_reader_with_file_path(Some(map_file.path()))
            .unwrap()
            .finish()
            .unwrap();

        let snps = df_map.shape().0;
        let mut ped_fields = vec![
            Field::new("family_id", DataType::String),
            Field::new("individual_id", DataType::Int32),
            Field::new("father_id", DataType::Int32),
            Field::new("mother_id", DataType::Int32),
            Field::new("sex", DataType::Int32),
            Field::new("affected_status", DataType::Int32),
        ];
        for x in 1..(snps + 1) {
            ped_fields.push(Field::new(format!("snp{}_1", x).as_str(), DataType::String));
            ped_fields.push(Field::new(format!("snp{}_2", x).as_str(), DataType::String));
        }
        let ped_schema = Schema::from_iter(ped_fields.into_iter());
        let ped_file_name = format!(
            "{}ped",
            map_file
                .file_name()
                .to_str()
                .unwrap()
                .trim_end_matches("map")
        );
        let ped_file = folder.join(&ped_file_name);
        let mut df_ped = CsvReadOptions::default()
            .with_has_header(false)
            .with_infer_schema_length(None)
            .with_schema(Some(Arc::new(ped_schema.clone())))
            .with_parse_options(CsvParseOptions::default().with_separator(b' '))
            .try_into_reader_with_file_path(Some(ped_file))
            .unwrap()
            .finish()
            .unwrap();

        df_ped
            .apply("family_id", |s: &Series| {
                s.str()
                    .unwrap()
                    .into_iter()
                    .map(|v: Option<&str>| v.map(|id: &str| format!("INCH_{}", id)))
                    .collect::<StringChunked>()
            })
            .unwrap();

        df_ped = df_ped
            .lazy()
            .with_row_index("index", Some(1))
            .collect()
            .unwrap();
        let mut p1 = df_ped
            .filter(
                df_ped
                    .column("index")
                    .unwrap()
                    .rem(2)
                    .equal(0)
                    .unwrap()
                    .as_ref(),
            )
            .unwrap();
        let part_1 = p1
            .apply("affected_status", |s: &Series| {
                s.i32()
                    .unwrap()
                    .into_iter()
                    .map(|v: Option<i32>| v.map(|_| 2))
                    .collect::<Int32Chunked>()
            })
            .unwrap();
        let mut p2 = df_ped
            .filter(
                df_ped
                    .column("index")
                    .unwrap()
                    .rem(2)
                    .equal(1)
                    .unwrap()
                    .bitand(df_ped.column("index").unwrap().not_equal(27).unwrap())
                    .as_ref(),
            )
            .unwrap();
        let part_2 = p2
            .apply("affected_status", |s: &Series| {
                s.i32()
                    .unwrap()
                    .into_iter()
                    .map(|v: Option<i32>| v.map(|_| 1))
                    .collect::<Int32Chunked>()
            })
            .unwrap();
        let part_3 = df_ped
            .filter(df_ped.column("index").unwrap().equal(27).unwrap().as_ref())
            .unwrap();
        let dd = concat(
            [part_1.clone().lazy(), part_2.clone().lazy(), part_3.lazy()],
            UnionArgs::default(),
        )
        .unwrap()
        .collect()
        .unwrap();
        df_ped = dd.sort(["index"], Default::default()).unwrap();

        df_map = df_map
            .lazy()
            .with_row_index("index", Some(1))
            .collect()
            .unwrap();
        let mut map_rows: Vec<LazyFrame> = vec![];
        for x in 1..(snps + 1) {
            let a = df_ped
                .column(format!("snp{}_1", x).as_str())
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .map(|v: Option<&str>| v.unwrap())
                .collect::<Vec<&str>>();
            let b = df_ped
                .column(format!("snp{}_2", x).as_str())
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .map(|v: Option<&str>| v.unwrap())
                .collect::<Vec<&str>>();
            let mut c: Vec<&str> = vec![];
            for y in 0..a.len() {
                c.push(a[y]);
                c.push(b[y]);
            }
            dbg!(&c);
            map_rows.push( df_map
                .filter(df_map.column("index").unwrap().equal(x as u64).unwrap().as_ref()).unwrap()
                .with_column(Series::new("snps", [c.join(" ")]))
                .unwrap()
                .to_owned().lazy());
            dbg!(&df_map);
        }
        let mr = concat(
            map_rows,
            UnionArgs::default(),
        )
            .unwrap()
            .collect()
            .unwrap();
        df_map = mr.sort(["index"], Default::default()).unwrap();

        df_map = df_map.drop("index").unwrap();
        df_ped = df_ped.drop("index").unwrap();
        create_dir_all(folder.join("output")).unwrap();
        let output_file = File::create(folder.join("output").join(ped_file_name)).unwrap();
        CsvWriter::new(output_file)
            .with_separator(b' ')
            .include_header(false)
            .finish(&mut df_ped)
            .unwrap();
        let map_output_file = File::create(folder.join("output").join(format!(
            "{}.tped",
            map_file.path().file_stem().unwrap().to_str().unwrap()
        )))
        .unwrap();
        CsvWriter::new(map_output_file)
            .with_separator(b' ')
            .include_header(false)
            .with_quote_style(QuoteStyle::Never)
            .finish(&mut df_map)
            .unwrap();
        let df_ped_transpose = File::create(folder.join("output").join(format!(
            "{}.tfam",
            map_file.path().file_stem().unwrap().to_str().unwrap()
        )))
        .unwrap();
        df_ped = df_ped
            .lazy()
            .select([cols([
                "family_id",
                "individual_id",
                "father_id",
                "mother_id",
                "sex",
                "affected_status",
            ])])
            .collect()
            .unwrap();
        CsvWriter::new(df_ped_transpose)
            .with_separator(b' ')
            .include_header(false)
            .finish(&mut df_ped)
            .unwrap();
    }
}
