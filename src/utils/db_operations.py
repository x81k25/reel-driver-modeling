# standard library imports
import os
import time

# third-party imports
from dotenv import load_dotenv
from loguru import logger
import polars as pl
import adbc_driver_postgresql.dbapi as adbc

# ------------------------------------------------------------------------------
# setup and config
# ------------------------------------------------------------------------------

# load dotenv at the module level if running locally
if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
    load_dotenv()

# ------------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------------

def gen_adbc_con(
    host: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_HOST'),
    port: str = int(os.getenv('REEL_DRIVER_TRNG_PGSQL_PORT', 5432)),
    dbname: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_DATABASE'),
    schema: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_SCHEMA'),
    user: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_USERNAME'),
    password: str = os.getenv('REEL_DRIVER_TRNG_PGSQL_PASSWORD')
) -> adbc.Connection:
    """
    function to create and return an ADBC connection object

    :param host: automatic-transmission database host
    :param port: automatic-transmission database port
    :param dbname: automatic-transmission database name
    :param schema: automatic-transmission schema name
    :param user: reel-driver service account username
    :param password: reel-driver service account password
    :return:
    """

    # ADBC uses URI-style connection strings
    uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    con = adbc.connect(uri)

    return con


con = gen_adbc_con()

# ------------------------------------------------------------------------------
# select functions
# ------------------------------------------------------------------------------

def select_star(
    table: str,
    schema: str = os.getenv("REEL_DRIVER_TRNG_PGSQL_SCHEMA")
) -> pl.DataFrame:
    """
    get the full training data table from pgsql

    :param table: string name of table to SELECT * FROM
    :param schema: database scheme to extract from
    :return: DataFrame of all needed training data
    """
    logger.info(f"starting data retrieval from {schema}.{table}")

    try:
        # Log connection attempt
        logger.debug("establishing database connection")
        con = gen_adbc_con()
        logger.debug("database connection established successfully")

        query = f"SELECT * FROM {schema}.{table}"
        logger.info(f"executing query: {query}")

        # Log before Polars read
        logger.debug("initiating Polars read_database operation")
        start_time = time.time()

        # Get schema information to handle problematic types systematically
        type_mapping = {}
        if "SELECT *" in query:
            # Get complete schema information for the table
            with con.cursor() as cur:
                cur.execute(f"""
                    SELECT column_name, data_type, ordinal_position
                    FROM information_schema.columns 
                    WHERE table_schema = '{schema}' AND table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                schema_info = cur.fetchall()
            
            # Build type mapping and identify problematic types
            problematic_types = {'numeric', 'decimal'}
            cast_columns = []
            regular_columns = []
            
            for col_name, data_type, position in schema_info:
                type_mapping[col_name] = data_type
                if data_type in problematic_types:
                    cast_columns.append(f"CAST({col_name} AS TEXT) AS {col_name}")
                else:
                    cast_columns.append(col_name)
                    regular_columns.append(col_name)
            
            # Reconstruct query with explicit casting for problematic types
            if any(data_type in problematic_types for _, data_type, _ in schema_info):
                query = f"SELECT {', '.join(cast_columns)} FROM {schema}.{table}"
                logger.debug(f"Modified query to handle problematic types: {query}")
                logger.debug(f"Type mapping: {type_mapping}")
        
        # Read the data with the modified query
        df = pl.read_database(
            query=query,
            connection=con
        )
        
        # Convert cast columns back to appropriate types based on original schema
        if type_mapping:
            for col_name, original_type in type_mapping.items():
                if col_name in df.columns and original_type in {'numeric', 'decimal'}:
                    logger.debug(f"Converting {col_name} from {original_type} (text) back to Float64")
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )

        # Log results and performance
        elapsed_time = time.time() - start_time
        logger.info(f"query completed in {elapsed_time:.2f} seconds")
        logger.info(
            f"retrieved {df.height} rows and {df.width} columns from {schema}.{table}")
        logger.debug(f"column names: {list(df.columns)}")
        logger.debug(f"dataframe shape: {df.shape}")
        logger.debug(f"memory usage: ~{df.estimated_size('mb'):.1f} MB")

    except Exception as e:
        logger.error(f"failed to retrieve data from {schema}.{table}: {str(e)}")
        raise
    finally:
        # Ensure connection is always closed
        try:
            con.close()
            logger.debug("database connection closed")
        except Exception as e:
            logger.warning(f"error closing database connection: {str(e)}")

    logger.info(f"data retrieval from {schema}.{table} completed successfully")
    return df


# ------------------------------------------------------------------------------
# insert functions
# ------------------------------------------------------------------------------

def trunc_and_load(
    df: pl.DataFrame,
    table_name: str,
    schema: str = os.getenv("REEL_DRIVER_TRNG_PGSQL_SCHEMA"),
    truncate: bool = True
):
    """
    Insert DataFrame rows into specified table.

    :param df: polars dataframe to be loaded
    :param table_name: name of table in the database to load to
    :param schema: database to schema to which table belongs
    :param truncate: whether or not to truncate the current able before
        inserting the new value
    :return: None
    """
    full_table_name = f"{schema}.{table_name}"
    row_count = df.height
    col_count = df.width

    logger.info(f"starting data load to {full_table_name}")
    logger.info(f"dataframe shape: {row_count} rows x {col_count} columns")
    logger.debug(f"columns to insert: {list(df.columns)}")
    logger.debug(f"truncate mode: {'enabled' if truncate else 'disabled'}")

    try:
        # Log connection attempt
        logger.debug("establishing database connection")
        con = gen_adbc_con()
        logger.debug("database connection established")

        start_time = time.time()

        with con.cursor() as cur:
            if truncate:
                logger.info(f"truncating table {full_table_name}")
                truncate_start = time.time()
                cur.execute(f"TRUNCATE TABLE {full_table_name};")
                truncate_time = time.time() - truncate_start
                logger.debug(
                    f"truncate completed in {truncate_time:.3f} seconds")

            # Auto-generate column list and placeholders
            columns = ", ".join(df.columns)
            placeholders = ", ".join(["$" + str(i+1) for i in range(len(df.columns))])
            logger.debug(f"preparing data conversion from DataFrame to tuples")

            data_conversion_start = time.time()
            data = [tuple(row) for row in df.iter_rows()]
            data_conversion_time = time.time() - data_conversion_start
            logger.debug(
                f"data conversion completed in {data_conversion_time:.3f} seconds")

            # Log insert operation
            logger.info(f"inserting {len(data)} rows into {full_table_name}")
            logger.debug(f"using executemany for batch insert")

            insert_start = time.time()
            cur.executemany(
                f"INSERT INTO {full_table_name} ({columns}) VALUES ({placeholders})",
                data
            )
            insert_time = time.time() - insert_start
            logger.debug(
                f"insert operation completed in {insert_time:.3f} seconds")

            # Log commit
            logger.debug("committing transaction")
            commit_start = time.time()
            con.commit()
            commit_time = time.time() - commit_start
            logger.debug(f"transaction committed in {commit_time:.3f} seconds")

        total_time = time.time() - start_time
        rows_per_second = row_count / total_time if total_time > 0 else 0

        logger.info(f"data load to {full_table_name} completed successfully")
        logger.info(f"total operation time: {total_time:.2f} seconds")
        logger.info(f"throughput: {rows_per_second:.0f} rows/second")

    except adbc.DatabaseError as e:
        logger.error(f"database error during load to {full_table_name}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"unexpected error during load to {full_table_name}: {str(e)}")
        raise
    finally:
        try:
            con.close()
            logger.debug("database connection closed")
        except Exception as e:
            logger.warning(f"error closing database connection: {str(e)}")


# ------------------------------------------------------------------------------
# end of db_operations.py
# ------------------------------------------------------------------------------