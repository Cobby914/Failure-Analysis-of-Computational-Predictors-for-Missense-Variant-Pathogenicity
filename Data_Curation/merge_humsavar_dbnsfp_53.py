import duckdb
import pandas as pd
import os
import glob
from tqdm import tqdm
import time


#########################################################
# FAST + PROGRESS BARS VERSION
#########################################################

if __name__ == "__main__":
    print("\n=======================================")
    print("   FAST HUMSAVAR + dbNSFP 5.3 MERGER")
    print("=======================================\n")

    start_time = time.time()

    #########################################################
    # LOAD HUMSAVAR
    #########################################################

    print("[INFO] Loading Humsavar...")
    humsavar = pd.read_csv("data/humsavar_labeled.csv", dtype=str)
    humsavar["dbSNP"] = humsavar["dbSNP"].astype(str).str.strip()

    HUMSAVAR_RS = set(humsavar["dbSNP"].unique())

    print(f"[INFO] Humsavar rows: {len(humsavar):,}")
    print(f"[INFO] Unique rsIDs: {len(HUMSAVAR_RS):,}")

    # Convert rsIDs to SQL tuple
    rs_list = "(" + ",".join(f"'{x}'" for x in HUMSAVAR_RS) + ")"


    #########################################################
    # LOCATE dbNSFP FILES
    #########################################################

    print("\n[INFO] Searching for dbNSFP 5.3 chromosome files...")
    db_path = r"C:\Users\Colin\dbNSFP5.3a"
    files = sorted(glob.glob(os.path.join(db_path, "dbNSFP5.3a_variant.chr*.gz")))

    if not files:
        print("[ERROR] No dbNSFP files found!")
        exit()

    print(f"[INFO] Found {len(files)} chromosome files.\n")


    #########################################################
    # BUILD UNION QUERY WITH PROGRESS BAR
    #########################################################

    print("[INFO] Building DuckDB streaming UNION plan...")
    union_sql_parts = []

    for f in tqdm(files, desc="Preparing file streams"):
        union_sql_parts.append(
            f"""
            SELECT *
            FROM read_csv(
                '{f}',
                delim='\t',
                header=True,
                nullstr='.',
                auto_detect=True,
                all_varchar=True
            )
            """
        )

    union_sql = "\nUNION ALL\n".join(union_sql_parts)


    #########################################################
    # EXECUTE MAIN STREAMING QUERY (PROGRESS SHOWN BY DuckDB)
    #########################################################

    print("\n[INFO] Running dbNSFP streaming merge...\n")

    con = duckdb.connect()
    con.register("hsv", humsavar)

    sql = f"""
        WITH db AS (
            {union_sql}
        )
        SELECT 
            hsv.Gene,
            hsv.Entry,
            hsv.FTId,
            hsv.AA_change,
            hsv.Category,
            hsv.dbSNP,
            hsv.Disease,
            hsv.Label,

            db."#chr" AS chr,
            db."pos(1-based)" AS pos,
            db.ref,
            db.alt,
            db.rs_dbSNP,

            db.SIFT_score,
            db.SIFT_pred,
            db.Polyphen2_HDIV_score,
            db.Polyphen2_HDIV_pred,
            db.CADD_raw,
            db.CADD_phred,
            db.REVEL_score

        FROM hsv
        LEFT JOIN db
        ON hsv.dbSNP = db.rs_dbSNP
        WHERE hsv.dbSNP IN {rs_list}
    """

    print("[INFO] DuckDB scan started... (this will show CPU/disk activity)")
    query_start = time.time()

    merged = con.execute(sql).fetchdf()

    query_end = time.time()
    print(f"[INFO] Streaming query finished in {query_end - query_start:.2f} seconds")


    #########################################################
    # SAVE FILE
    #########################################################

    out_file = "data/humsavar_dbnsfp53_merged_fast.csv"
    merged.to_csv(out_file, index=False)

    print(f"\n[INFO] Final merged rows: {len(merged):,}")
    print("[INFO] Saved →", out_file)

    total_time = time.time() - start_time
    print(f"\n[DONE] Pipeline completed in {total_time/60:.2f} minutes.\n")
