import scanpy as sc
import gspread
import pandas as pd
import numpy as np
import anndata as ad
from rich import print
from rich.pretty import pprint
from typing import Optional

def get_gspread_df(
    credential_json: str,
    gspread_name: str,
    metadata_type: str = 'Tier_1',
    sheet_name: str = 'obs'
    ) -> pd.DataFrame:
    """
    Get a DataFrame from a Google Spreadsheet.

    Parameters
    ----------
    credential_json
        The path to the JSON file containing the credentials.
    gspread_name
        The name of the Google Spreadsheet.
    sheet_name
        The name of the sheet to read from.

    Returns
    -------
    A pandas DataFrame.
    """
    metadata_type = metadata_type.capitalize()
    sheet_name = sheet_name.lower()

    valid_metadata_types = {'Tier_1', 'Tier_2'}
    valid_sheet_names = {'obs', 'uns'}

    if metadata_type not in valid_metadata_types:
        raise ValueError(f'Not a valid metadata_type. Please specify one of {valid_metadata_types}')
    if sheet_name not in valid_sheet_names:
        raise ValueError(f'Not a valid sheet name.Please specify one of {valid_sheet_names}')

    sa = gspread.service_account(filename=credential_json)
    sheet = sa.open(gspread_name)
    work_sheet = sheet.worksheet(f"{metadata_type}_{sheet_name}")
    df = pd.DataFrame(work_sheet.get_all_records())
    if sheet_name == 'obs':
        df = df.iloc[4:, 1:].reset_index(drop=True)
    if sheet_name == 'uns':
        df = df.iloc[5:, 1:].reset_index(drop=True)

    return df

class MetadataValidator:
    def __init__(
        self,
        df
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Provided data must be a pandas DataFrame")
        self.df = df

    def validate(
        self,
        df_type
    ):
        if df_type == "uns":
            required_keys = ['title', 'study_PI', 'unpublished']
            return self._validate_keys(required_keys)
        elif df_type == "obs":
            return self._validate_obs()
        else:
            raise ValueError("Invalid df_type specified. Choose 'uns' or 'obs'")

    def _validate_keys(self, required_keys):
        missing_keys = [key for key in required_keys if key not in self.df.columns]
        if missing_keys:
            return False, {"Missing keys": missing_keys}

        invalid_entries = {
            key: "Each entry must be a non-empty string."
            for key in required_keys
            if self.df[key].isnull().any()
                or (self.df[key].apply(lambda x: not isinstance(x, str) or not x.strip())).any()
        }

        if invalid_entries:
            return False, invalid_entries

        return True, "Validation successful: All required keys are present with valid string entries."

    def _validate_obs(self):
        required_columns = [
            'sample_ID', 'donor_id', 'protocol_URL', 'institute', 'sample_collection_site', 
            'sample_collection_relative_time_point', 'library_ID', 'library_ID_repository', 
            'author_batch_notes', 'organism_ontology_term_id', 'manner_of_death', 'sample_source', 
            'sex_ontology_term_id', 'sample_collection_method', 'tissue_type', 'sampled_site_condition', 
            'tissue_ontology_term_id', 'tissue_free_text', 'sample_preservation_method', 'suspension_type', 
            'cell_enrichment', 'cell_viability_percentage', 'cell_number_loaded', 'sample_collection_year', 
            'assay_ontology_term_id', 'library_preparation_batch', 'library_sequencing_run', 
            'sequenced_fragment', 'sequencing_platform', 'is_primary_data', 'reference_genome', 
            'gene_annotation_version', 'alignment_software', 'intron_inclusion', 'disease_ontology_term_id', 
            'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id'
        ]

        errors = {}
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            errors["Missing columns"] = missing_columns

        bool_columns = ['is_primary_data']
        for column in bool_columns:
            self.df[column] = self.df[column].str.lower()

        # Specific requirements for each field with explicit checks
        checks = {
            'organism_ontology_term_id': (
                "Must be 'NCBITaxon:9606' for human samples.",
                lambda x: all(item.strip() == "NCBITaxon:9606" for item in x.split(','))
            ),
            'manner_of_death': (
                "Must be one of {1, 2, 3, 4, 0, 'unknown', 'not applicable'}.",
                lambda x: all(item.strip() in {'1', '2', '3', '4', '0', 'unknown', 'not applicable'} for item in x.split(',')) if isinstance(x, str)
                else (x in {1, 2, 3, 4, 0} if isinstance(x, int) else all(item in {1, 2, 3, 4, 0} for item in x) if isinstance(x, (list, tuple, set)) else False)
            ),
            'sample_source': (
                "Must be one of {'surgical donor', 'postmortem donor', 'organ donor'}.",
                lambda x: all(item.strip() in {'surgical donor', 'postmortem donor', 'organ donor'} for item in x.split(','))
            ),
            'sex_ontology_term_id': (
                "Must be one of {'PATO:0000383', 'PATO:0000384'} for male or female.",
                lambda x: all(item.strip() in {'PATO:0000383', 'PATO:0000384'} or item.strip() in {'na', 'NA', 'N/A', 'nan'} for item in x.split(','))
            ),
            'sample_collection_method': (
                "Must be one of {'biopsy', 'brush', 'surgical resection'}.",
                lambda x: all(item.strip() in {'biopsy', 'brush', 'surgical resection'} for item in x.split(','))
            ),
            'tissue_type': (
                "Must be one of {'tissue', 'organoid', 'cell culture'}.",
                lambda x: all(item.strip() in {'tissue', 'organoid', 'cell culture'} for item in x.split(','))
            ),
            'sampled_site_condition': (
                "Must be one of {'healthy', 'diseased', 'adjacent'}.",
                lambda x: all(item.strip() in {'healthy', 'diseased', 'adjacent'} for item in x.split(','))
            ),
            'tissue_ontology_term_id': (
                "Must start with 'UBERON:' or be 'na', 'NA', or 'N/A'.",
                lambda x: all(item.strip().startswith('UBERON:') or item.strip() in {'na', 'NA', 'N/A', 'nan'} for item in x.split(','))
            ),
            'sample_preservation_method': (
                "Must be a valid method such as 'fresh', 'frozen at -80C', etc.",
                lambda x: all(item.strip() in {
                    'ambient temperature', 'cut slide', 'fresh', 'frozen at -70C', 'frozen at -80C', 'frozen at -150C',
                    'frozen in liquid nitrogen', 'frozen in vapor phase', 'paraffin block', 'RNAlater at 4C', 'RNAlater at 25C',
                    'RNAlater at -20C', 'other'
                }
                for item in x.split(','))
            ),
            'suspension_type': (
                "Must be one of {'cell', 'nucleus', 'na'}.",
                lambda x: all(item.strip() in {'cell', 'nucleus', 'na', 'nan'} for item in x.split(','))
            ),
            'assay_ontology_term_id': (
                "Must start with 'EFO' or be 'na', 'NA', or 'N/A'.",
                lambda x: all(item.strip().startswith('EFO') or item.strip() in {'na', 'NA', 'N/A', 'nan'} for item in x.split(','))
            ),
            'sequenced_fragment': (
                "Must be one of {'3 prime tag', '3 prime end bias', '5 prime tag', '5 prime end bias', 'full length'}.",
                lambda x: all(item.strip() in {
                    '3 prime tag', '3 prime end bias', '5 prime tag', '5 prime end bias', 'full length'
                }
                or item.strip() in {'na', 'NA', 'N/A'} for item in x.split(','))
            ),
            'sequencing_platform': (
                "Must start with 'EFO' or be 'na', 'NA', or 'N/A'.",
                lambda x: all(item.strip().startswith('EFO') or item.strip() in {'na', 'NA', 'N/A', 'nan'} for item in x.split(','))
            ),
            'is_primary_data': (
                "Must be 'true' or 'false'.",
                lambda x: all(item.strip() in {'true', 'false'} for item in x.split(','))
            ),
            'reference_genome': (
                "Must be one of {'GRCh38', 'GRCh37', 'GRCm39', 'GRCm38', 'GRCm37', 'not applicable'}.",
                lambda x: all(item.strip() in {'GRCh38', 'GRCh37', 'GRCm39', 'GRCm38', 'GRCm37', 'not applicable'} for item in x.split(','))
            ),
            'intron_inclusion': (
                "Must be 'yes' or 'no'.",
                lambda x: all(item.strip() in {'yes', 'no'} for item in x.split(','))
            ),
            'disease_ontology_term_id': (
                "Must start with 'PATO:' or 'MONDO:'.",
                lambda x: all(item.strip().startswith('PATO:') or item.strip().startswith('MONDO:') for item in x.split(','))
            ),
            'development_stage_ontology_term_id': (
                "Must start with 'HsapDv:' or be 'na', 'NA', or 'N/A'.",
                lambda x: all(item.strip().startswith('HsapDv:') or item.strip() in {'na', 'NA', 'N/A'} for item in x.split(','))
            )
        }

        # additional checks for columns that require all entries to be non-null strings
        string_columns = [
            'sample_ID', 'donor_id', 'protocol_URL', 'institute', 'sample_collection_site',
            'sample_collection_relative_time_point', 'library_ID', 'library_ID_repository',
            'author_batch_notes', 'gene_annotation_version', 'alignment_software',
            'self_reported_ethnicity_ontology_term_id'
        ]

        for column in string_columns:
            if column in self.df.columns and not self.df[column].apply(lambda x: isinstance(x, str) or x in {'NA', 'na', 'N/A', 'nan'} or pd.isna(x)).all():
                errors[column] = "All entries must be non-null strings."

        # checks for columns that allow numeric values or 'NA'
        numeric_or_na_columns = ['cell_viability_percentage', 'cell_number_loaded', 'sample_collection_year']
        for column in numeric_or_na_columns:
            if column in self.df.columns and not self.df[column].apply(lambda x: isinstance(x, (int, float, np.int64, np.float64)) or x in {'NA', 'na', 'N/A', 'nan'} or pd.isna(x)).all():
                errors[column] = "Entries must be integers, floats, or 'NA'/'na'/'N/A'."

        # run checks
        for column, (message, check) in checks.items():
            if column in self.df.columns:
                invalid_entries = ~self.df[column].apply(check)
                if invalid_entries.any():
                    errors[column] = f"{message} Invalid entries found."

        if errors:
            print("Validation error(s) encountered.")
            pprint(errors)
            return False, errors
        return True, "Validation successful: All checks on DataFrame passed."

class ShowErrorsFromValidator:
    def __init__(
        self,
        errors_dict: dict
    ) -> None:
        if not isinstance(errors_dict, dict):
            raise ValueError("Provided error log is not a dict.")
        self.errors = errors_dict

    def show_errors(
        self,
        obs_df: pd.DataFrame
    ) -> None:
        for key, val in self.errors.items():
            print(
                f"Error containing field: {key}\n"
                f"Error message: {val}\n"
                f"Value counts:\n"
                f"{obs_df[key].value_counts()}"
            )

class RemoveListCharacters:
    def __init__(
        self
    ):
        pass

    def remove_characters(self, df):
        df.replace(to_replace="[\\[\\]'\"]", value='', regex=True, inplace=True)
        df.replace(to_replace=";", value=',', regex=True, inplace=True)
        return df

class ValidationWorkflow:
    def __init__(
        self,
        input,
        axis: str
    ) -> None:

        if isinstance(input, ad.AnnData):
            if axis == 'obs':
                obs_columns = [
                    'sample_ID', 'donor_id', 'protocol_URL', 'institute', 'sample_collection_site', 
                    'sample_collection_relative_time_point', 'library_ID', 'library_ID_repository', 
                    'author_batch_notes', 'organism_ontology_term_id', 'manner_of_death', 'sample_source', 
                    'sex_ontology_term_id', 'sample_collection_method', 'tissue_type', 'sampled_site_condition', 
                    'tissue_ontology_term_id', 'tissue_free_text', 'sample_preservation_method', 'suspension_type', 
                    'cell_enrichment', 'cell_viability_percentage', 'cell_number_loaded', 'sample_collection_year', 
                    'assay_ontology_term_id', 'library_preparation_batch', 'library_sequencing_run', 
                    'sequenced_fragment', 'sequencing_platform', 'is_primary_data', 'reference_genome', 
                    'gene_annotation_version', 'alignment_software', 'intron_inclusion', 'disease_ontology_term_id', 
                    'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id'
                ]
                valid_columns = input.obs.columns.intersection(obs_columns)
                valid_obs_df = input.obs[valid_columns].copy()
                self.input = valid_obs_df.astype(str)
            if axis == 'uns':
                uns_keys = ['title', 'study_PI', 'batch_condition', 'default_embedding', 'unpublished', 'comments']
                uns_dict = {key: input.uns[key] for key in uns_keys if key in input.uns}
                self.input = pd.DataFrame([uns_dict]).astype(str)

        if isinstance(input, pd.DataFrame):
            self.input = input

        self.axis = axis

    def init_workflow(
        self
        ) -> pd.DataFrame:

        # remove list characters
        remover = RemoveListCharacters()
        updated_df = remover.remove_characters(self.input)

        # perform actual validation
        validator = MetadataValidator(updated_df)
        validation_out = validator.validate(df_type=self.axis)

        if validation_out[0]:
            print(f"Validation workflow successful. Returning {self.axis.upper()} dataframe.")
        else:
            error_shower = ShowErrorsFromValidator(validation_out[1])
            error_shower.show_errors(updated_df)
            print(f"Returning error-containing {self.axis.upper()} dataframe.")

        return updated_df

class AnnDataMerger:
    def __init__(
        self,
        adata,
        obs_df: Optional[pd.DataFrame] = None,
        uns_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the AnnDataMerger with an AnnData object and a DataFrame.

        Parameters
        ----------
        adata (AnnData): The AnnData object containing single-cell data.
        df (DataFrame): A pandas DataFrame containing metadata.
        """
        self.adata = adata
        self.obs_df = obs_df
        self.uns_df = uns_df

    def validate_and_prepare_data(
        self,
        df_col
    ) -> None:
        """
        Validates and prepares the DataFrame by setting a unique index for merging.

        Parameters
        ----------
        df_col (str): Column name in DataFrame to set as index.
        """
        if self.obs_df[df_col].is_unique:
            self.obs_df.set_index(df_col, inplace=True)
        else:
            raise ValueError(f"The column {df_col} must have unique values to serve as an index for mapping.")

    def add_obs_metadata(
        self,
        adata_col,
        df_col,
        skip: Optional[list]
    ) -> ad.AnnData:
        """
        Adds metadata OBS from the DataFrame to the AnnData object's .obs by matching columns.

        Parameters
        ----------
        adata_col (str): Column name in self.adata.obs to match on.
        df_col (str): Column name in self.df to match on.

        Returns
        -------
        AnnData: The AnnData object with updated metadata.
        """
        self.validate_and_prepare_data(df_col)
        if skip is None:
            skip = []
        for col in self.obs_df.columns:
            if col not in skip:
                if col in self.adata.obs.columns:
                    print(f"Warning: Column {col} already exists in AnnData OBS. Overwriting with new values.")
                self.adata.obs[col] = self.adata.obs[adata_col].map(self.obs_df[col])

        # Change sample_id in adata to sample_ID (corresponds to HCA tier 1 metadata template)
        if adata_col == 'sample_id':
            self.adata.obs.rename(columns={'sample_id': 'sample_ID'}, inplace=True)

        return self.adata

    def add_uns_metadata(
        self
    ) -> ad.AnnData:
        """
        Adds UNS metadata from the DataFrame to the AnnData object's .uns.

        Parameters
        ----------
        None.

        Returns
        -------
        AnnData: The AnnData object with updated metadata.
        """
        uns_dict = self.uns_df.to_dict()
        uns_dict_fil = {out_key: in_val for out_key, out_dict in uns_dict.items() for in_key, in_val in out_dict.items()}

        for key, val in uns_dict_fil.items():
            if key in self.adata.uns.keys():
                print(f"Warning: Key {key} already exists in AnnData UNS. Overwriting with new values.")
            self.adata.uns[key] = val

        return self.adata
