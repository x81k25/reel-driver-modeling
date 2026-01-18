# standard library imports
from pathlib import Path

# third-party imports
from loguru import logger
import polars as pl
import re
import yaml

# custom/local imports
import src.utils as utils

# -----------------------------------------------------------------------------
# config loading
# -----------------------------------------------------------------------------

def load_production_company_config() -> dict:
    """
    Load production company normalization config from YAML.

    :return: dict containing normalization_rules and encoding_settings
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "production_companies.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_company_normalization_map(config: dict) -> dict:
    """
    Build a reverse lookup map from variant names to canonical names.

    :param config: loaded YAML config dict
    :return: dict mapping lowercase variant names to canonical names
    """
    normalize_map = {}
    for canonical, variants in config['normalization_rules'].items():
        for variant in variants:
            normalize_map[variant.lower().strip()] = canonical
    return normalize_map


def normalize_production_companies(
    series: pl.Series,
    normalize_map: dict
) -> pl.Series:
    """
    Normalize production company names in a Series of lists.

    :param series: pl.Series containing lists of company names
    :param normalize_map: dict mapping variant names to canonical names
    :return: pl.Series with normalized company names
    """
    def normalize_list(companies):
        if companies is None:
            return None
        normalized = []
        for company in companies:
            key = company.lower().strip()
            normalized.append(normalize_map.get(key, company))
        return normalized

    return series.map_elements(normalize_list, return_dtype=pl.List(pl.String))


# -----------------------------------------------------------------------------
# supporting functions
# -----------------------------------------------------------------------------

def one_hot_encode_list_column(
    series: pl.Series,
    column_name: str,
    max_categories: int | None = None
) -> tuple[list[str], list[list[int]]]:
    """
	One-hot encode a Polars Series containing lists of strings.

	:param series: pl.Series containing lists of strings to be one-hot encoded
	:param column_name: str, original column name to use as prefix for generated column names
	:param max_categories: optional int, limit encoding to top N most frequent values
	:return: tuple containing list of column names and list of one-hot encoded arrays
	"""
    from collections import Counter

    def clean_value(value: str) -> str:
        # Replace non-word characters with underscores and convert to lowercase
        cleaned = re.sub(r'[^\w]', '_', value.lower())
        # Replace multiple consecutive underscores with single underscore
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        return cleaned.strip('_')

    # Handle null/empty cases
    if series.is_null().all():
        return [column_name], [[0] for _ in range(len(series))]

    # Get all values across all lists with their frequencies
    all_values_list = []
    for row in series:
        if row is not None and len(row) > 0:
            all_values_list.extend(row)

    # If no values found, return original column name
    if not all_values_list:
        return [column_name], [[0] for _ in range(len(series))]

    # Count frequencies and optionally limit to top N
    value_counts = Counter(all_values_list)
    if max_categories is not None:
        # Get top N most frequent values
        top_values = [v for v, _ in value_counts.most_common(max_categories)]
        sorted_values = sorted(top_values)
        logger.info(f"{column_name}: limited to top {max_categories} of {len(value_counts)} unique values")
    else:
        sorted_values = sorted(value_counts.keys())

    column_names = [f"{column_name}_{clean_value(value)}" for value in
                    sorted_values]

    # Create set for O(1) lookup
    values_set = set(sorted_values)

    # Create one-hot encoded arrays
    encoded_arrays = []
    for row in series:
        if row is None or len(row) == 0:
            encoded_arrays.append([0] * len(sorted_values))
        else:
            # Only encode values that are in our selected set
            encoded_row = [1 if value in row else 0 for value in sorted_values]
            encoded_arrays.append(encoded_row)

    return column_names, encoded_arrays


# -----------------------------------------------------------------------------
# transform
# -----------------------------------------------------------------------------

def filter_training(training: pl.DataFrame) -> pl.DataFrame:
    """
    get the full training data table from pgsql

    :param training: input DataFrame
    :return: output DataFrame with transformation complete
    """
    training_filtered = training.clone()

    logger.info(f"performing filtering on {training_filtered.height} row of training data")

    training_filtered = (
        training_filtered
            # we are currently only analyzing move data
            .filter(pl.col('media_type') == 'movie')
            # pull in only training data that has been confirmed for accuracy
            .filter(pl.col('reviewed'))
            # select only field which will actually be used
            .select([
                # identifiers
                'imdb_id',
                'media_title',
                # label
                'label',
                # continuous
                'release_year',
                'budget',
                'revenue',
                'runtime',
                'tmdb_rating',
                'tmdb_votes',
                'rt_score',
                'metascore',
                'imdb_rating',
                'imdb_votes',
                # categorical
                'origin_country',
                'production_status',
                # categorical lists
                'production_companies',
                'production_countries',
                'original_language',
                'spoken_languages',
                'genre',
                # flags
                'anomalous',
            ])
    )

    logger.info(f"returning {training_filtered.height} rows filtered data")

    return training_filtered


def num_training(training: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    perform transformation on all numeric fields

    :param training: input training DataFrame
    :return: tuple containing [
        DataFrame with properly processed numeric fields,
        DataFrame of the range of values to be used for future normalizations
    ]
    """
    logger.info("performing basic numerical transformation on training")

    training_num = training.clone()

    num_columns = [
        "release_year",
        "budget",
        "revenue",
        "runtime",
        "tmdb_rating",
        "tmdb_votes",
        "rt_score",
        "metascore",
        "imdb_rating",
        "imdb_votes"
    ]

    # create dataframe to hold min_max values
    # instantiate DataFrame to hold column_name_mappings
    normalization_table = pl.DataFrame(schema={
        "feature": pl.String,
        "min": pl.Float64,
        "max": pl.Float64
    })

    for num_column in num_columns:
        # get normalized values and add to normalization_table
        col_min = float(training_num.select(pl.col(num_column).drop_nulls().min())[0, 0])
        col_max = float(training_num.select(pl.col(num_column).drop_nulls().max())[0, 0])

        schema_row = pl.DataFrame({
            'feature': num_column,
            'min': col_min,
            'max': col_max
        })
        normalization_table = pl.concat([normalization_table, schema_row])

        # get normalized values
        training_num = training_num.with_columns(
            (pl.col(num_column) - pl.lit(col_min)) /
            (pl.lit(col_max) - pl.lit(col_min))
            .alias(num_column)
        )

    logger.info("basic numerical transformation complete")

    return training_num, normalization_table


def cat_training(training: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    perform transformation on all categorical fields

    :param training: input training DataFrame
    :return: DataFrame with properly processed categorical fields
    :return: DataFrame containing the column_name mappings for sparse matrices
        which will be stored as vectors inside of the primary DataFrame
    """
    logger.info("performing categorical operations on training data")

    training_cat = training.clone()

    # instantiate DataFrame to hold column_name_mappings
    engineered_schema = pl.DataFrame(schema={
        "original_column": pl.String,
        "exploded_mapping": pl.List(pl.String)
    })

    # load production company config and normalize names
    company_config = load_production_company_config()
    normalize_map = build_company_normalization_map(company_config)
    max_companies = company_config['encoding_settings']['max_categories']

    training_cat = training_cat.with_columns(
        normalize_production_companies(
            training_cat['production_companies'],
            normalize_map
        ).alias('production_companies')
    )

    logger.info(f"production_companies normalized, encoding top {max_companies}")

    # columns to one-hot encode without category limits
    list_columns = [
        'origin_country',
        'production_countries',
        'spoken_languages',
        'genre'
    ]

    # for each categorical fields
    #   explode column and create sparse matrices
    #   store column mappings in schema DataFrame
    #   store column value as an array in the original column of the data frame
    for list_column in list_columns:
        # encode values
        names, values = one_hot_encode_list_column(
            series=training_cat[list_column],
            column_name=list_column
        )

        # Add as new column
        training_cat = (
                training_cat.with_columns(
                pl.Series(
                    (list_column + '_encoded'), values
                )
            ).drop(list_column)
            .rename({(list_column + '_encoded'): list_column})
        )

        schema_row = pl.DataFrame({
            'original_column': [list_column],
            'exploded_mapping': [names]
        })
        engineered_schema = pl.concat([engineered_schema, schema_row])

    # encode production_companies with max_categories limit
    names, values = one_hot_encode_list_column(
        series=training_cat['production_companies'],
        column_name='production_companies',
        max_categories=max_companies
    )

    training_cat = (
        training_cat.with_columns(
            pl.Series('production_companies_encoded', values)
        ).drop('production_companies')
        .rename({'production_companies_encoded': 'production_companies'})
    )

    schema_row = pl.DataFrame({
        'original_column': ['production_companies'],
        'exploded_mapping': [names]
    })
    engineered_schema = pl.concat([engineered_schema, schema_row])

    logger.info("categorical operations complete")

    return training_cat, engineered_schema


def encode_training(training: pl.DataFrame) -> pl.DataFrame:
    """
    encoded training label for model ingestion based on relevant featurs

    :param training: input training DataFrame
    :return: DataFrame with training label encoded
    """
    logger.info("encoding training label")

    training_encoded = training.clone()

    training_encoded = training_encoded.with_columns(
        label=pl.when(pl.col('label') == 'would_watch')
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )

    logger.info("training label encoded")

    return training_encoded


# -----------------------------------------------------------------------------
# primary function
# -----------------------------------------------------------------------------

def xgb_prep():
    """
    full pipeline to transform training data into a fully engineered feature
        set ready for ingestion and storage back into the database
    """

    # get full data set
    training = utils.select_star(table="training")

    # perform preliminary polars operations
    engineered = filter_training(training)

    # perform numeric transformation
    engineered, normalization_table = num_training(engineered)

    # perform categorical transformations
    engineered, engineered_schema = cat_training(engineered)

    # encode training label
    engineered = encode_training(engineered)

    # print summary information
    logger.info(engineered.group_by('label').agg(pl.len()))
    logger.info(engineered.select([
        'budget',
        'revenue',
        'runtime',
        'release_year',
        'tmdb_rating',
        'tmdb_votes',
        'rt_score',
        'metascore',
        'imdb_rating',
        'imdb_votes'
    ]).describe())

    # store all engineered training values and metadata
    utils.trunc_and_load(
        df=engineered,
        table_name="engineered"
    )
    utils.trunc_and_load(
        df=normalization_table,
        table_name="engineered_normalization_table"
    )
    utils.trunc_and_load(
        df=engineered_schema,
        table_name="engineered_schema"
    )


# main guard
def __main__():
    xgb_prep()

# ------------------------------------------------------------------------------
# end of _02_xgb_binomial_classifier_prep_data.py
# ------------------------------------------------------------------------------