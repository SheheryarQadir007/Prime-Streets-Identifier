import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Deflactor:
    def __init__(self, df, start_date_column, end_date_column, zone_column, price_column, area_column, save_df_as_reference):
        """
        Initialize the Deflactor class with the required data.
        :param df: DataFrame containing the ad data.
        :param start_date_column: Column containing the start date information (i.e: 'first_appearance').
        :param end_date_column: Column containing the end date information (i.e: 'last_appearance').
        :param price_column: Column containing the price information (i.e: 'local_price').
        :param area_column: Column containing the area information (i.e: 'area').
        :param zone_column: Optional, column to subgroup by (e.g., zona).
        """
        self.df = df
        self.start_date_column = start_date_column
        self.end_date_column = end_date_column
        self.zone_column = zone_column
        self.price_column = price_column
        self.area_column = area_column
        self.save_df_as_reference = save_df_as_reference


    def preprocess_data(self):
        """
        Drop rows with:
        - 'area' below 20
        - 'local_price' to 'area' ratio greater than 25,000
        - Null ('NaN') values in 'local_price' or 'area'
        - 'local_price' equal to 0
        """
        print("Preprocessing data: Dropping rows with invalid 'area', 'local_price', or zero prices...")
        
        self.df = self.df.copy()

        # Drop rows where 'local_price' or 'area' is NaN
        self.df = self.df.dropna(subset=[self.price_column, self.area_column])
        
        # Drop rows where 'local_price' is 0
        self.df = self.df[self.df[self.price_column] > 0]

        # Calculate the price-to-area ratio
        self.df.loc[:, 'price_m2'] = self.df[self.price_column] / self.df[self.area_column]

        # Apply filtering conditions
        self.df = self.df[
            (self.df[self.area_column] >= 20) & 
            (self.df['price_m2'] <= 25000)
        ]
        
        print(f"Data after preprocessing: {len(self.df)} rows remain.")




    def expand_ad_months(self):
        # ---------> Before, the fucntion was not getting cases where First and Last appearance were in the same month
        """Expand the data to include all months between the start and end appearance dates."""
        expanded_data = []
        print(f"Expanding data for {len(self.df)} rows...")

        for _, row in self.df.iterrows():
            # Generate all months between first_appearance and last_appearance
            all_months = pd.date_range(row[self.start_date_column], row[self.end_date_column], freq='MS').strftime('%Y-%m')
            if all_months.empty:
                # Temporarily convert string dates to datetime
                first_appearance = pd.to_datetime(row[self.start_date_column])
                all_months = [first_appearance.strftime('%Y-%m')]

            # Add a row for each month
            for month in all_months:
                expanded_data.append({
                    self.zone_column : row[self.zone_column],
                    'year_month': pd.to_datetime(month, format='%Y-%m'), # ||||||This was changed to remain in Datetime format|||||
                    'local_price': row[self.price_column],
                    'area': row[self.area_column],
                })

        expanded_df = pd.DataFrame(expanded_data)
        print(f"Expanded data has {len(expanded_df)} rows.")
        return expanded_df


    def generate_complete_combinations(self, expanded_df):
        """Generate a DataFrame with all combinations of 'zona' and 'year_month'."""
        print(f"Generating complete combinations of zona and year_month...")

        # Generate all unique zones and months
        all_zonas = expanded_df[self.zone_column].unique()
        all_months = pd.date_range(                 #  ||||||This was changed to work with the updated Datetime format|||||
            start=expanded_df['year_month'].min(),
            end=expanded_df['year_month'].max(),
            freq='MS'
        )    

        # Create a DataFrame with all combinations of zona and year_month
        complete_combinations = pd.MultiIndex.from_product([all_zonas, all_months], names=[self.zone_column, 'year_month'])
        complete_df = pd.DataFrame(index=complete_combinations).reset_index()

        print(f"Generated complete combinations (first 5 rows): \n{complete_df.head()}")
        return complete_df
    


    def merge_and_sort(self, expanded_df, complete_df):
        """
        Merge the expanded data with the complete combinations and sort by zona and year_month,
        dynamically handling zones with insufficient data points in the last month.
        """
        print(f"Merging data...")

        # Aggregate the data by zona and year_month
        aggregated_data = expanded_df.groupby(
            [self.zone_column, 'year_month']
        )[[self.price_column, self.area_column]].mean().reset_index()

        print(f"Aggregated data (first 5 rows): \n{aggregated_data.head()}")

        # Step 2: Handle zones with insufficient data points in the last month
        # Count rows in each zona and year_month
        counts = expanded_df.groupby([self.zone_column, 'year_month']).size().reset_index(name='counts')

        # Merge the counts back into the aggregated data
        aggregated_data = pd.merge(aggregated_data, counts, on=[self.zone_column, 'year_month'], how='left')

        # Identify zones with 2 or fewer data points in the last month
        insufficient_data = aggregated_data[aggregated_data['counts'] <= 2]

        for zona in insufficient_data[self.zone_column].unique():
            zona_data = aggregated_data[aggregated_data[self.zone_column] == zona]

            # Get the last month for this zona
            last_month = zona_data['year_month'].max()

            # Filter data for the last 3 months including the last
            recent_months = zona_data[zona_data['year_month'] >= (pd.Timestamp(last_month) - pd.DateOffset(months=2))]

            # Compute weighted moving average for the last 3 months
            weights = range(1, len(recent_months) + 1)  # Weights: 1, 2, 3 for recent months
            weighted_price = np.average(recent_months[self.price_column], weights=weights)
            weighted_area = np.average(recent_months[self.area_column], weights=weights)

            # Update the last month's price and area with the weighted averages
            aggregated_data.loc[
                (aggregated_data[self.zone_column] == zona) & (aggregated_data['year_month'] == last_month),
                [self.price_column, self.area_column]
            ] = [weighted_price, weighted_area]

        # Step 4: Merge the aggregated data with the complete combinations
        final_df = pd.merge(complete_df, aggregated_data, on=[self.zone_column, 'year_month'], how='left')
        
        print(f"Merged data (first 5 rows): \n{final_df.head()}")
        return final_df


    def fill_missing_values(self, final_df):
        """Fill missing values for price and area using nearest values."""
        print(f"Filling missing values...")
        
        # Check missing values before and after filling
        print("Missing values before filling:")
        print(final_df.isnull().sum())

        # Initialize global counters
        global_counts = {'filled': 0, 'unfilled': 0}

        def fill_nearest(series):
            """Helper function to fill missing values using the nearest values."""
            filled = series.copy()
            for idx in series[series.isna()].index:
                before = series.loc[:idx].last_valid_index()
                after = series.loc[idx:].first_valid_index()

                if before is not None and after is not None:
                    filled[idx] = (series[before] + series[after]) / 2
                    global_counts['filled'] += 1
                elif before is not None:
                    filled[idx] = series[before]
                    global_counts['filled'] += 1
                elif after is not None:
                    filled[idx] = series[after]
                    global_counts['filled'] += 1
                else:
                    global_counts['unfilled'] += 1

            return filled

        # Apply the logic to each column
        if self.area_column:
            final_df[self.area_column] = final_df.groupby(self.zone_column, group_keys=False)[self.area_column].apply(fill_nearest)

        if self.price_column:
            final_df[self.price_column] = final_df.groupby(self.zone_column, group_keys=False)[self.price_column].apply(fill_nearest)

        # Recalculate or average price_m2
        final_df['avg_price_m2'] = final_df[self.price_column] / final_df[self.area_column]

        print("Missing values after filling:")
        print(final_df.isnull().sum())

        # Check specific zones for 2018-10
        print(final_df[(final_df[self.zone_column] == 'G3 - Goya') & (final_df['year_month'] == '2018-10')])

        # Print groups with missing values
        for zona, group in final_df.groupby(self.zone_column):
            if group.isnull().any().any():
                print(f"Missing values in zona: {zona}")
                print(group[group.isnull().any(axis=1)])


        # Global summary
        total_missing = global_counts['filled'] + global_counts['unfilled']
        print(f"Global Summary: {total_missing} missing values found, {global_counts['filled']} filled, {global_counts['unfilled']} could not be filled.")

        return final_df

    
    def calculate_inflation_factor(self, final_df):
        """
        Calculate the inflation factor for each row based on the last available price per square meter (avg_price_m2) for each zona.
        """
        print(f"Calculating inflation factor based on price per square meter (avg_price_m2)...")

        # Find the most recent avg_price_m2 for each zona
        most_recent_avg_price_m2 = (
            final_df
            .loc[final_df.groupby(self.zone_column)['year_month'].idxmax()]
            .set_index(self.zone_column)['avg_price_m2']
            .rename('most_recent_avg_price_m2')
        )
        print(most_recent_avg_price_m2.head())

        # Merge the most recent avg_price_m2 back into the original DataFrame
        final_df = final_df.merge(most_recent_avg_price_m2, on=self.zone_column, how='left')

        # Calculate the inflation factor based on avg_price_m2
        final_df['inflation_factor'] = final_df['most_recent_avg_price_m2'] / final_df['avg_price_m2']

        # Drop the temporary column 'most_recent_avg_price_m2' if not needed
        #final_df.drop(columns=['most_recent_avg_price_m2'], inplace=True)

        print(f"Calculated inflation factors (first 5 rows): \n{final_df[['inflation_factor', 'avg_price_m2']].head()}")
        return final_df

    
    def detect_correct_outliers(self, final_df, z_threshold=3):
        """
        Detect, correct, and validate outliers in the dataset based on inflation_factor.
        
        Changes:
        - Works with year_month-level aggregated data for neighbor handling.
        - Corrects outliers using aggregated averages of 2 previous and 2 next months when necessary.
        - Adds a new column 'inflation_factor_corrected' to final_df.
        
        :param final_df: DataFrame to process.
        :param z_threshold: Threshold for z-score to classify an outlier.
        :return: Updated DataFrame with corrected inflation_factor.
        """
        print("Detecting and correcting outliers...")

        # Step 1: Aggregate inflation_factor by year_month and zona
        aggregated = (
            final_df.groupby([self.zone_column, 'year_month'])['inflation_factor']
            .mean()
            .reset_index()
            .rename(columns={'inflation_factor': 'avg_inflation_factor'})
        )

        # Step 2: Process each zona independently
        def process_group(group):
            # Sort by year_month to ensure chronological order
            group = group.sort_values(by='year_month').reset_index(drop=True)

            # Prepare corrected values
            corrected_values = []

            for idx, row in group.iterrows():
                year_month = row['year_month']
                avg_inflation_factor = row['avg_inflation_factor']

                # Filter data for the current year_month
                current_month_data = final_df[
                    (final_df[self.zone_column] == row[self.zone_column]) & (final_df['year_month'] == year_month)
                ]

                if len(current_month_data) > 3:  # Enough data points
                    # Calculate z-scores for the current year_month
                    mean = current_month_data['inflation_factor'].mean()
                    std = current_month_data['inflation_factor'].std()
                    current_month_data['z_score'] = (current_month_data['inflation_factor'] - mean) / std

                    # Replace outliers with the mean of non-outlier values
                    non_outliers = current_month_data[
                        current_month_data['z_score'].abs() <= z_threshold
                    ]['inflation_factor']
                    corrected_value = non_outliers.mean()
                else:  # Not enough data points
                    # Handle sparse year_month with neighbors
                    neighbors = pd.concat([
                        group.iloc[max(0, idx - 2): idx],  # Previous 2 months
                        group.iloc[idx + 1: idx + 3]      # Next 2 months
                    ])
                    if not neighbors.empty:
                        corrected_value = neighbors['avg_inflation_factor'].mean()
                    else:
                        # Fallback to original value
                        corrected_value = avg_inflation_factor

                corrected_values.append(corrected_value)

            # Add corrected values to the group
            group['avg_inflation_factor_corrected'] = corrected_values
            return group

        # Apply the correction process to each zona
        corrected_aggregated = aggregated.groupby(self.zone_column, group_keys=False).apply(process_group)

        # Step 3: Merge corrected aggregated values back into the original final_df
        final_df = pd.merge(
            final_df,
            corrected_aggregated[[self.zone_column, 'year_month', 'avg_inflation_factor_corrected']],
            on=[self.zone_column, 'year_month'],
            how='left'
        )


        def calculate_weighted_moving_average(series, window=3):
            """
            Calculate the weighted moving average for a given series.
            Weights are 1, 2, ..., window size.
            """
            weights = range(1, window + 1)
            return series.rolling(window=window, min_periods=1).apply(
                lambda x: np.dot(x, weights[-len(x):]) / sum(weights[-len(x):]), raw=True
            )

        final_df['wma_inflation_factor'] = final_df.groupby(self.zone_column, group_keys=False)[
            'avg_inflation_factor_corrected'
        ].apply(lambda group: calculate_weighted_moving_average(group, window=3))

        print("Outliers detected, corrected, and WMA applied.")
        return final_df


    def Normalized_prices(self, final_df):
      """Calculate the normalized prices based on the inflation factor."""
      print(f"Normalizing prices...")
      final_df['normalized_price'] = final_df[self.price_column] / final_df['wma_inflation_factor']

      print(f"Normalized prices (first 5 rows): \n{final_df[['normalized_price', self.price_column]].head()}")
      return final_df
    
    def create_Reference(self, final_df):
        print("Entered into Create reference to upload on GCP.")
        if self.save_df_as_reference:
            reference_df = final_df[[self.zone_column, 'year_month', 'wma_inflation_factor', self.area_column,self.price_column]]
            print(f"Inflation DataFrame prepared for saving (first 5 rows):\n{reference_df.head()}")
            print("Uploading the DataFrame to GCP for future reference (placeholder).")
            # Placeholder for actual GCP upload logic
            return reference_df
        
    def Normalizing_Single_value(self, zone, year_month):
        if self.reference_df is None:
            print("There is no data frame that matches users request")
            return None
        
        matching_rows = self.reference_df[
            (self.reference_df[self.zone_column] == zone) & 
            (self.reference_df['year_month'] == year_month)
        ]
        
        if not matching_rows.empty:
            wma_inflation_factor = matching_rows['wma_inflation_factor'].iloc[0]
            price = matching_rows['local_price'].iloc[0]
            
            print(f"Inflation factor for {zone} in {year_month} whose price {price} was: The Inflation Factor is : {wma_inflation_factor}")
            
            normalized_price = price / wma_inflation_factor
            print(f"Normalized price for {zone} in {year_month} whose price {price} was: The Normalized Price is:{normalized_price} ")
            return normalized_price
        
        print(f"No matching inflation factor found for {zone} in {year_month}")
        return None
    


    def add_deflated_price(self, original_df, processed_df):
        """
        Add a deflated_price column to the original dataset using inflation factors
        from the processed dataset.
        """
        print("Adding deflated_price column to the original dataset...")

        # Convert first_appearance to datetime and extract year_month
        original_df[self.start_date_column] = pd.to_datetime(original_df[self.start_date_column], errors='coerce')
        original_df['year_month'] = pd.to_datetime(original_df[self.start_date_column]).dt.to_period('M').dt.to_timestamp()


        # Merge wma_inflation_factor from processed_df into original_df based on zona and year_month
        merged_df = pd.merge(
            original_df,
            processed_df[[self.zone_column, 'year_month', 'wma_inflation_factor']],  # Only wma_inflation_factor is needed
            on=[self.zone_column, 'year_month'],
            how='left'
        )

        # Calculate deflated_price as local_price * wma_inflation_factor
        merged_df['deflated_price'] = merged_df['local_price'] * merged_df['wma_inflation_factor']

        print(f"Updated dataset (first 5 rows):\n{merged_df.head()}")

        return merged_df
    


    def deflactor(self):
        """Full processing pipeline for ad data."""
        print("Starting full processing pipeline...")

    	# Preprocess data
        self.preprocess_data()

        expanded_df = self.expand_ad_months()

        # Generate the complete combinations of zona and year_month
        complete_df = self.generate_complete_combinations(expanded_df)

        # Merge expanded_df with complete_df based on zona and year_month
        final_df = self.merge_and_sort(expanded_df, complete_df)

        # Fill missing values in the final_df
        final_df = self.fill_missing_values(final_df)

        # Calculate the inflation factor for the final_df
        final_df = self.calculate_inflation_factor(final_df)

        # Detect and correct outliers in the inflation_factor
        final_df = self.detect_correct_outliers(final_df)

        # # Normalize prices in the final_df
        final_df = self.Normalized_prices(final_df)

        # #Check wether the Reference should be added to GCP or Not
        reference_df = self.create_Reference(final_df)

        self.reference_df = reference_df
        
        # # Integrate deflated_price back into the original dataset
        self.df = self.add_deflated_price(self.df, final_df)

        if reference_df is not None:
            print("Reference DataFrame is ready for further processing or upload.")
        else:
            print("No reference DataFrame created.")

        print("Pipeline completed.")
        return self.df