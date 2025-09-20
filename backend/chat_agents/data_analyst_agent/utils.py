import re
import logging 
import json 
import traceback 
import pandas as pd 
import copy 
import numpy as np 

def extract_json(input_str: str):
    """
    Extracts the last complete JSON object found within curly braces {} 
    from an input string. Uses a greedy regex match for the initial block.
    Attempts to fix common issues like trailing commas, single quotes for strings/keys,
    and Python-style booleans/None. It will also attempt to parse only the first
    JSON object if trailing non-JSON data is present after fixes.
    """
    # Pattern to find a block enclosed in curly braces.
    # The original function used a greedy match to find the last such block.
    pattern = r'(\{.*\})' 
    matches = re.findall(pattern, input_str, re.DOTALL)
    
    if not matches:
        # Handle cases where no {}-enclosed block was found at all
        return "Pattern '{}' not found in the input string."

    # Use the last match found (as per the original logic)
    match_content = matches[-1].strip()

    # Step 1: Get rid of trailing commas before a closing brace or bracket.
    clean_match = re.sub(r',\s*([}\]])', r'\1', match_content)
    
    try:
        # Attempt to parse the initially cleaned JSON string directly
        json_dict = json.loads(clean_match)
        return json_dict
        
    except json.JSONDecodeError as e_initial:
        # If direct parsing fails, attempt to fix common LLM-generated issues.
        try:
            # Start with the comma-cleaned string
            fixed_str = clean_match
            
            # Step 2: Replace Pythonic None, True, False with JSON null, true, false.
            # This is done *before* quote changes for unquoted keywords.
            fixed_str = re.sub(r'\bNone\b', 'null', fixed_str)
            fixed_str = re.sub(r'\bTrue\b', 'true', fixed_str)
            fixed_str = re.sub(r'\bFalse\b', 'false', fixed_str)
            
            # Step 3: Convert single-quoted strings/keys to double-quoted strings/keys.
            fixed_str = re.sub(r"'((?:\\.|[^'])*)'", r'"\1"', fixed_str)
            
            # Step 4: Attempt to parse the fixed string using raw_decode.
            # This will parse the first valid JSON object and allow for ignoring
            # subsequent non-JSON text (like LLM "thinking" text).
            decoder = json.JSONDecoder()
            json_dict, _ = decoder.raw_decode(fixed_str) # We don't strictly need the end_index here
            return json_dict # Successfully extracted the JSON object
            
        except json.JSONDecodeError as e_fixed_final:
            # This exception means that fixed_str is malformed even for raw_decode.
            # This could be due to issues within the JSON structure itself, not just trailing data.
            error_context_length = 75 

            # Details from the initial parsing attempt (e_initial on clean_match)
            original_error_pos = e_initial.pos
            original_doc = e_initial.doc 
            original_start = max(0, original_error_pos - error_context_length)
            original_end = min(len(original_doc), original_error_pos + error_context_length)
            original_snippet = original_doc[original_start:original_end]
            original_relative_pos = original_error_pos - original_start
            original_marked_snippet = original_snippet[:original_relative_pos] + "[ERROR->" + original_snippet[original_relative_pos:]

            # Details from the final parsing attempt (e_fixed_final on fixed_str)
            fixed_error_pos = e_fixed_final.pos
            fixed_doc = e_fixed_final.doc 
            fixed_start = max(0, fixed_error_pos - error_context_length)
            fixed_end = min(len(fixed_doc), fixed_error_pos + error_context_length)
            fixed_snippet = fixed_doc[fixed_start:fixed_end]
            fixed_relative_pos = fixed_error_pos - fixed_start
            fixed_marked_snippet = fixed_snippet[:fixed_relative_pos] + "[ERROR->" + fixed_snippet[fixed_relative_pos:]

            return (f"Initial JSON parsing failed: {e_initial.msg} at position {e_initial.pos}. "
                    f"Problematic text (original): ...{original_marked_snippet}...\n"
                    f"Attempted fixes (quotes, keywords), but parsing the fixed string also failed: {e_fixed_final.msg} at position {e_fixed_final.pos}. "
                    f"Problematic text (after fixes): ...{fixed_marked_snippet}...\n"
                    f"Original string (after trailing comma removal) snippet: {clean_match[:250]}...\n"
                    f"String after attempted fixes snippet: {fixed_str[:250]}...")

def extract_python(input_str):
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, input_str, re.DOTALL)

    return matches[0]

def convert_to_features_list(dataframes_dict):    
    features_list = {}
    for filename, item in dataframes_dict.items():
        if isinstance(item, pd.DataFrame):
            features_list[filename] = list(dataframes_dict[filename].columns)

        elif isinstance(item, dict):
            features_list[filename] = {}
            for pagename in dataframes_dict[filename]:
                features_list[filename][pagename] = list(dataframes_dict[filename][pagename].columns)

        else:
           features_list[filename] = {} 

    features_list_json = extract_json(str(features_list))
    
    return features_list_json


def execute_code(code, local_var): 
    logger = logging.getLogger('error_logger')
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler('error_log_data_analyst_agent.txt')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)

    try:
        namespace = copy.deepcopy(local_var)
        exec(code, namespace)
        result = namespace['main']()
        success = True
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        error_message = "Traceback:\n"
        for frame in tb:
            filename, line_number, function_name, text = frame
            error_message += f"File: {filename}, Line: {line_number}, in {function_name}\n"
            error_message += f"  {text}\n"
        error_message += f"Error: {e}"

        # logging error
        logger.error(error_message)                    
        result = error_message
        success = False
        
    return result, success
    




def standardize_file(df_input, default_year=2025, **kwargs):
    """
    Standardizes a DataFrame by:
    1. Uppercasing and replacing spaces with underscores in column names.
    2. Identifying a potential date/time column.
    3. Converting and standardizing this date/time column to 'YYYY-MM-DD' format,
       naming it 'DATE', and moving it to the front.
       - If a 'MONTH' column (with month names) is found, it uses this with a 'YEAR'
         column (if available) or default_year, defaulting to the 1st day of the month.
       - Otherwise, it attempts to parse the identified time column using pd.to_datetime
         and formats valid dates to 'YYYY-MM-DD'.
    4. Standardizing other string column values (uppercase, strip, replace space with underscore).

    Args:
        df_input (pd.DataFrame): The input DataFrame.
        default_year (int, optional): The default year to use if a year column is not found
                                      or contains missing values when processing a 'MONTH' column.
                                      Defaults to 2025.
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        pd.DataFrame: The standardized DataFrame.
                      Returns an empty DataFrame if the input is empty.

    Raises:
        TypeError: If df_input is not a pandas DataFrame.
    """

    POSSIBLE_TIME_COLUMNS = [
        'DATE', 'MONTH', 'TIME', 'PERIOD', 'YEARMONTH', 'DATETIME', 'TIMESTAMP',
        'DATE_TIME', 'DT', 'TRANS_DT', 'EVENT_DATE', 'ACTIVITY_DATE'
    ]
    MONTH_MAP = {
        'JAN': 1, 'JANUARY': 1, 'FEB': 2, 'FEBRUARY': 2, 'MAR': 3, 'MARCH': 3,
        'APR': 4, 'APRIL': 4, 'MAY': 5, 'JUN': 6, 'JUNE': 6, 'JUL': 7, 'JULY': 7,
        'AUG': 8, 'AUGUST': 8, 'SEP': 9, 'SEPT': 9, 'SEPTEMBER': 9, 'OCT': 10, 'OCTOBER': 10,
        'NOV': 11, 'NOVEMBER': 11, 'DEC': 12, 'DECEMBER': 12
    }

    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df_input.empty:
        return pd.DataFrame() # Return empty DataFrame if input is empty

    df = df_input.copy() # Work on a copy

    # --- 1. Standardize Column Names ---
    try:
        df.columns = [str(col).upper().strip().replace(' ', '_') for col in df.columns]
        standardized_cols = df.columns
    except Exception as e:
        # print(f"Warning: Could not standardize column names: {e}") # Optional logging
        standardized_cols = df.columns # Use original names if standardization fails

    # --- 2. Identify and Process Potential Date Column ---
    time_col_name = next((col for col in POSSIBLE_TIME_COLUMNS if col in standardized_cols), None)
    processed_date_col_name = None # Track if date processing was successful
    new_date_col = 'DATE'          # Target name for the processed date column

    if time_col_name:
        try:
            original_dtype = df[time_col_name].dtype
            temp_standardized_dates_str = pd.Series(index=df.index, dtype=str) # Store 'YYYY-MM-DD' strings

            is_likely_month_col = False
            if time_col_name == 'MONTH' and pd.api.types.is_string_dtype(original_dtype):
                try:
                    unique_vals = df[time_col_name].dropna().astype(str).str.upper().unique()
                    if any(m in MONTH_MAP for m in unique_vals[:10]):
                        is_likely_month_col = True
                except Exception:
                    pass # Ignore errors during this heuristic check

            if is_likely_month_col:
                # --- Handle Month Name Column Case (e.g., 'MONTH' column with 'JAN', 'FEB') ---
                month_nums = df[time_col_name].astype(str).str.upper().map(MONTH_MAP)
                base_valid_idx = month_nums.notna() # Rows where month name was successfully mapped

                # Determine the year to use for each row
                year_values_for_construction = pd.Series(index=df.index, dtype='object')
                actual_default_year = default_year if default_year is not None else pd.Timestamp.now().year

                if 'YEAR' in standardized_cols:
                    year_series_numeric = pd.to_numeric(df['YEAR'], errors='coerce')
                    year_values_for_construction = year_series_numeric.fillna(actual_default_year)
                else:
                    year_values_for_construction.fillna(actual_default_year, inplace=True)

                # Ensure both month and year are valid for construction
                # and year is a whole number (int)
                final_valid_idx = base_valid_idx & \
                                  year_values_for_construction.notna() & \
                                  (year_values_for_construction.apply(lambda x: isinstance(x, (int, float)) and x == int(x)))


                if final_valid_idx.any():
                    years_str = year_values_for_construction[final_valid_idx].astype(int).astype(str)
                    months_str = month_nums[final_valid_idx].astype(int).astype(str).str.zfill(2)
                    day_str = "01" # Default to the 1st day of the month

                    temp_standardized_dates_str.loc[final_valid_idx] = years_str + '-' + months_str + '-' + day_str
            
            else:
                # --- Handle Other Potential Date/Time Column Cases ---
                # Attempt standard datetime conversion
                # Pandas to_datetime can infer many formats. errors='coerce' turns unparseable dates into NaT.
                datetime_col = pd.to_datetime(df[time_col_name], errors='coerce')
                valid_idx = datetime_col.notna() # Find where conversion succeeded
                
                if valid_idx.any():
                    # Format valid datetime objects to 'YYYY-MM-DD' string
                    temp_standardized_dates_str.loc[valid_idx] = datetime_col[valid_idx].dt.strftime('%Y-%m-%d')

            # --- Assign results to DataFrame and clean up ---
            if not temp_standardized_dates_str.isnull().all():
                df[new_date_col] = temp_standardized_dates_str.replace({np.nan: None})
                processed_date_col_name = new_date_col

                if time_col_name != new_date_col and time_col_name in df.columns:
                    df = df.drop(columns=[time_col_name])
                
                # Move the new/processed 'DATE' column to the front
                cols = [processed_date_col_name] + [col for col in df.columns if col != processed_date_col_name]
                df = df[cols]
            else:
                if new_date_col in df.columns and df[new_date_col].isnull().all():
                    df = df.drop(columns=[new_date_col], errors='ignore')

        except Exception as e:
            # print(f"Warning: Could not process time column '{time_col_name}': {e}") # Optional logging
            processed_date_col_name = None # Ensure it's marked as failed

    # --- 3. Standardize String Column Values ---
    try:
        string_cols = df.select_dtypes(include=['object', 'string']).columns

        if processed_date_col_name and processed_date_col_name in string_cols:
            # Check dtype just in case it was converted to something else unexpectedly
            # The 'DATE' column should now contain strings like 'YYYY-MM-DD' or None
            if pd.api.types.is_object_dtype(df[processed_date_col_name].dtype) or \
               pd.api.types.is_string_dtype(df[processed_date_col_name].dtype):
                string_cols = string_cols.drop(processed_date_col_name)

        for col in string_cols:
            if col in df.columns: # Check column still exists
                # Ensure we only apply string methods to actual string data, handling NaNs
                # Convert to string, then apply operations. NaNs become 'nan' string.
                # Strip whitespace, uppercase, replace space with underscore.
                # df.loc[:, col] = df[col].astype(str).str.strip().str.upper().str.replace(' ', '_')
                
                # More robust handling for mixed types / NaNs before string conversion
                mask_notna = df[col].notna()
                df.loc[mask_notna, col] = df.loc[mask_notna, col].astype(str).str.strip().str.upper().str.replace(' ', '_')


    except Exception as e:
        # print(f"Warning: Could not standardize string column values: {e}") # Optional logging
        pass

    # --- 4. Return Standardized DataFrame ---
    return df


def convert_features_list_to_array(features_list):
    collected_words_set = set()

    def extract_all_strings(element):
        """
        Recursively extracts all strings from a nested structure,
        capitalizes them, and adds them to collected_words_set.
        """
        if isinstance(element, str):
            collected_words_set.add(element.upper())
        elif isinstance(element, list):
            for item in element:
                extract_all_strings(item) # Recurse for each item in the list
        elif isinstance(element, dict):
            for key, value in element.items():
                collected_words_set.add(key.upper()) # Add dictionary key
                extract_all_strings(value)      # Recurse for dictionary value

    # Start the extraction process with your main data structure
    extract_all_strings(features_list)

    # Convert the set to a list
    result_array = list(collected_words_set)
    return result_array