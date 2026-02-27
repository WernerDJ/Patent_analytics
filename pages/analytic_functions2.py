"""
Created on Tue Apr  1 11:57:44 2025

@author: werner
"""
from django.conf import settings #Necessary to pull the OpenAI API Key
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
plt.rcParams['font.family'] = 'DejaVu Sans'
import regex as re
from collections import Counter
import numpy as np
from datetime import datetime  
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from wordcloud import WordCloud
import networkx as nx
import io
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz, process
import tempfile
from openai import OpenAI
import tiktoken
import sys #just for debugging

#Import English stopwords from the nltk repository downloaded to the hard drive
nltk.data.path.append("/usr/share/nltk_data")


#Divide the word tokens into the three main syntactic groups
pos_groups = {
    'Nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
    'Verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'Adjectives': ['JJ', 'JJR', 'JJS']
}

class Patent_Analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_excel(filepath, skiprows=5)
        # For debugging the excel download only
        #import sys
        #sys.stderr.write(f">>> DataFrame columns: {self.data.columns.tolist()}\n")
        #sys.stderr.write(f">>> First few rows:\n{self.data.head().to_string()}\n")
        #sys.stderr.flush()
        self.data['MainIPC'] = self.data['I P C'].apply(self.extract_main_ipc)
        self.data['PubYear'] = pd.to_datetime(
            self.data['Publication Date'], format='%d.%m.%Y', errors='coerce'
        ).dt.year
        self.filtered_data = self.data.copy()

    
    def load_stopwords(self, input_txt_path='pages/stopwords.txt'):
        stopwords_set = set(nltk.corpus.stopwords.words('english'))
        try:
            with open(input_txt_path, "r", encoding="utf-8") as f:
                include_stopwords = {line.strip() for line in f}
            stopwords_set |= include_stopwords
        except FileNotFoundError:
            print(f"Warning: Stopwords file not found at {input_txt_path}")
        return stopwords_set

    def plot_top_10_countries(self):
        if self.filtered_data  is None:
            raise ValueError("Data not loaded. Please load the Excel file first.")
        
        # Exclude EP and WO
        country_counts = self.filtered_data [~self.filtered_data ['Country'].isin(['EP', 'WO'])]['Country'].value_counts()
        top_countries = country_counts.head(10)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        top_countries.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Top 10 Most Frequent Publication Countries (Excluding EP and WO)")
        ax.set_xlabel("Country")
        ax.set_ylabel("Number of Publications")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        return fig
 
    def extract_priority_info(self, priority_data):
        """Extract the earliest priority year and corresponding country from priority data."""
        if pd.isna(priority_data):
            return None, None

        earliest_year = None
        earliest_country = None
        entries = priority_data.split(';')

        for entry in entries:
            entry = entry.strip()
            match = re.search(r'(?:\S+\s+)?(\d{2}\.\d{2}\.(\d{4}))\s+(\w+)', entry)
            if match:
                year = int(match.group(2))
                country = match.group(3)
            else:
                match = re.search(r'(?:\S+\s+)?(\d{4})-(\d{2})-(\d{2})\s+(\w+)', entry)
                if match:
                    year = int(match.group(1))
                    country = match.group(4)
                else:
                    match = re.search(r'.*?(\d{4})[-./].*?\s+(\w+)$|.*?(\w+)\s+(\d{4})[-./]', entry)
                    if match:
                        if match.group(1) and match.group(2):
                            year = int(match.group(1))
                            country = match.group(2)
                        elif match.group(3) and match.group(4):
                            country = match.group(3)
                            year = int(match.group(4))
                    else:
                        continue

            if year is not None and (earliest_year is None or year < earliest_year):
                earliest_year = year
                earliest_country = country

        return earliest_year, earliest_country

    def prepare_priority_data(self):
        """Prepare DataFrame with one row per unique patent family, containing earliest year and country."""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            raise ValueError("No data available for analysis")

        priority_groups = self.filtered_data.groupby('Priorities Data')
        priority_info = []

        for _, group in priority_groups:
            priority_data = group['Priorities Data'].iloc[0]
            year, country = self.extract_priority_info(priority_data)
            if year is not None and country is not None:
                priority_info.append((year, country))

        return pd.DataFrame(priority_info, columns=['Year', 'Country'])

    def plot_priority_countries_bar(self, priority_df):
        """Plot a bar chart of the Top 10 Priority Countries (deduplicated families)."""
        top_countries = priority_df['Country'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_countries.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Top 10 Priority Countries")
        ax.set_xlabel("Country")
        ax.set_ylabel("Number of Priorities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        return fig

    def plot_priority_years_bar(self, priority_df):
        """Plot a bar chart of frequency of priorities by year (deduplicated families)."""
        year_counts = priority_df.groupby('Year').size().sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        year_counts.plot(kind='bar', color='lightgreen', ax=ax)
        ax.set_title("Priority Years Frequency")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Priorities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        return fig

    
    def analyze_patent_flow(self, top_n=10):
        """
        Analyzes patent flow between origin countries (from Priorities Data) and 
        destination countries (from Country column).
        
        - Extracts priority country codes from Priorities Data
        - Ignores rows where priority country equals publication country
        - Ignores rows where publication country or origin country is WO or EP
        - Creates a flow diagram showing patent movement from top N origin countries to destination countries
        
        Args:
            top_n: Number of top origin countries to include in the visualization (default: 10)
        """
        # Create a copy of filtered data to work with
        df = self.filtered_data.copy()
        
        # Extract the origin country (priority country) from Priorities Data column
        def extract_priority_country(priorities_data):
            if pd.isna(priorities_data) or priorities_data == '':
                return None
            
            # Extract all country codes from priorities data
            # Format is typically: "number date country; number date country"
            matches = re.findall(r'[0-9]+ [0-9\.]+\s+([A-Z]{2})', str(priorities_data))
            if matches:
                return matches[0]  # Return the first priority country code
            return None
        
        # Apply the function to extract origin countries
        df['Origin'] = df['Priorities Data'].apply(extract_priority_country)
        
        # Rename the Country column to Destination for clarity
        df['Destination'] = df['Country']
        
        # Filter the data according to requirements
        filtered_df = df[
            (df['Origin'].notna()) &  # Priority data is not empty
            (df['Origin'] != df['Destination']) &  # Priority country is different from publication country
            (~df['Origin'].isin(['WO', 'EP'])) &  # Origin country is not WO or EP
            (~df['Destination'].isin(['WO', 'EP']))  # Publication country is not WO or EP
        ]
        
        # Get the top N origin countries by total count
        origin_total_counts = filtered_df['Origin'].value_counts().head(top_n)
        top_origins = origin_total_counts.index.tolist()
        
        # Filter data to include only top origin countries
        filtered_df = filtered_df[filtered_df['Origin'].isin(top_origins)]
        
        # Count the frequency of each origin-destination pair
        flow_df = filtered_df.groupby(['Origin', 'Destination']).size().reset_index(name='Count')
        
        # Count the total occurrences of each origin and destination country
        origin_counts = flow_df.groupby('Origin')['Count'].sum().reset_index()
        dest_counts = flow_df.groupby('Destination')['Count'].sum().reset_index()
        
        # Create the visualization
        figure = self._create_patent_flow_diagram(flow_df, origin_counts, dest_counts, top_n)
        
        return figure
    
    def _create_patent_flow_diagram(self, flow_df, origin_counts, dest_counts, top_n=10):
        """
        Creates a flow diagram showing patent movement from origin to destination countries.
        
        Args:
            flow_df: DataFrame with Origin, Destination, and Count columns
            origin_counts: DataFrame with count of patents per origin country
            dest_counts: DataFrame with count of patents per destination country
            top_n: Number of top origin countries included in the analysis
        """
        # Create a figure with a single axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Import required libraries if not already imported
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        # Define colors for countries
        unique_countries = pd.concat([
            pd.Series(flow_df['Origin'].unique()), 
            pd.Series(flow_df['Destination'].unique())
        ]).unique()
        
        # Create a color map for countries
        color_map = {}
        cmap = plt.cm.tab20
        for i, country in enumerate(unique_countries):
            color_map[country] = cmap(i % 20)
        
        # Set up the positions for the bars
        left_x = 0
        right_x = 1
        
        # Sort countries by frequency
        origin_counts = origin_counts.sort_values('Count', ascending=False)
        dest_counts = dest_counts.sort_values('Count', ascending=False)
        
        # Calculate the total for normalization
        total_origin = origin_counts['Count'].sum()
        total_dest = dest_counts['Count'].sum()
        
        # Prepare the positions and heights for the origin bars
        origin_positions = {}
        y_offset = 0
        for _, row in origin_counts.iterrows():
            country = row['Origin']
            count = row['Count']
            height = count / total_origin
            origin_positions[country] = (y_offset, height)
            
            # Draw the origin bar
            ax.add_patch(plt.Rectangle(
                (left_x, y_offset), 
                0.1, 
                height, 
                color=color_map[country], 
                alpha=0.8
            ))
            
            # Add label for origin country - moved outside the bar
            if height > 0.01:
                ax.text(
                    left_x - 0.01, 
                    y_offset + height/2, 
                    country, 
                    ha='right', 
                    va='center', 
                    fontsize=9
                )
                
            y_offset += height
        
        # Prepare the positions and heights for the destination bars
        dest_positions = {}
        y_offset = 0
        for _, row in dest_counts.iterrows():
            country = row['Destination']
            count = row['Count']
            height = count / total_dest
            dest_positions[country] = (y_offset, height)
            
            # Draw the destination bar
            ax.add_patch(plt.Rectangle(
                (right_x - 0.1, y_offset), 
                0.1, 
                height, 
                color=color_map[country], 
                alpha=0.8
            ))
            
            # Add label for destination country - moved outside the bar
            if height > 0.01:
                ax.text(
                    right_x + 0.01, 
                    y_offset + height/2, 
                    country, 
                    ha='left', 
                    va='center', 
                    fontsize=9
                )
            
            y_offset += height
        
        # Draw the connecting lines between origin and destination
        for _, row in flow_df.iterrows():
            origin = row['Origin']
            dest = row['Destination']
            count = row['Count']
            
            # Get the positions
            origin_y = origin_positions[origin][0] + origin_positions[origin][1] / 2
            dest_y = dest_positions[dest][0] + dest_positions[dest][1] / 2
            
            # Calculate the line width based on the count
            line_width = 1 + 5 * (count / flow_df['Count'].max())
            
            # Calculate the transparency based on the count
            alpha = 0.2 + 0.6 * (count / flow_df['Count'].max())
            
            # Define the curve using control points
            verts = [
                (left_x + 0.1, origin_y),                   # Start point
                ((left_x + right_x) / 2, origin_y),         # Control point 1
                ((left_x + right_x) / 2, dest_y),           # Control point 2
                (right_x - 0.1, dest_y)                     # End point
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            
            # Draw the path with the appropriate color and width
            patch = patches.PathPatch(
                path, 
                facecolor='none',
                edgecolor=color_map[origin],
                linewidth=line_width,
                alpha=alpha,
                zorder=1
            )
            ax.add_patch(patch)
        
        # Adjust the x limits to accommodate the labels
        ax.set_xlim(left_x - 0.1, right_x + 0.1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add labels for the bars
        ax.text(left_x + 0.05, 1.05, 'Origin Countries', ha='center', va='bottom', fontsize=12)
        ax.text(right_x - 0.05, 1.05, 'Destination Countries', ha='center', va='bottom', fontsize=12)
        
        # Add title above the graph
        plt.figtext(0.5, 0.95, f"Destination Countries of the top {top_n} priority countries", 
                     ha='center', fontsize=14, weight='bold')
        
        # Remove axis frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the title
        return fig


    def extract_main_ipc(self, ipc_codes):
        if pd.isna(ipc_codes):
            return None
        codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipc_codes))
        if not codes:
            return None
        short_codes = [code[:4] for code in codes]
        counter = Counter(short_codes)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else short_codes[0]

    def filter_by_ipc_and_year(self, ipc_list=None, year_range=None):
        data = self.filtered_data
        if ipc_list:
            data = self.data[data['MainIPC'].isin(ipc_list)]
        if year_range and len(year_range) == 2:
            start, end = year_range
            data = data[(data['PubYear'] >= start) & (data['PubYear'] <= end)]
        self.filtered_data = data
        return data

    def plot_top_ipcs(self, top_n=10):
        all_ipcs = []
        for ipcs in self.filtered_data['I P C'].dropna():
            codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipcs))
            all_ipcs.extend(codes)
        ipc_counts = pd.Series(all_ipcs).value_counts().head(top_n)
        self.top_ipc_counts = ipc_counts  # Store for reuse in the ellaboration of a boring table
        plt.figure(figsize=(10, 6))
        ipc_counts.plot(kind='bar', color='lightgreen')
        plt.title(f"Top {top_n} Most Frequent IPCs")
        plt.xlabel("IPC Code")
        plt.ylabel("Number of Publications")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
        

    def get_top_ipcs_with_titles(self, title_file="cpc_group_titles.csv"):
        """
        Uses stored top IPC counts and enriches them with titles from the CPC definition CSV.
        Then creates and saves a PNG image containing the table.
        """
        if not hasattr(self, 'top_ipc_counts'):
            raise ValueError("Top IPCs not calculated yet. Please call plot_top_ipcs() first.")
    
        # Load IPC titles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        title_path = os.path.join(current_dir, title_file)
    
        if not os.path.isfile(title_path):
            raise FileNotFoundError(f"Title file '{title_file}' not found at {title_path}.")
    
        df_titles = pd.read_csv(title_path)
        title_dict = dict(zip(df_titles['Group'], df_titles['Title']))
    
        # Create a new figure without using matplotlib's table
        fig, ax = plt.subplots(figsize=(12, 0.5 * (len(self.top_ipc_counts) + 1)))
        ax.axis('off')
        
        # Remove all margins and make axes fill figure
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Create our own custom table layout
        header = ["IPC Group", "Count", "Title"]
        rows = []
        for ipc in self.top_ipc_counts.index:
            count = int(self.top_ipc_counts[ipc])
            title = title_dict.get(ipc, "No title found")
            rows.append([ipc, count, title])
        
        # Calculate positions
        row_height = 0.8 / (len(rows) + 1)
        font_size = 10
        
        # Draw header
        for i, col_name in enumerate(header):
            x_pos = 0.02 if i == 0 else (0.17 if i == 1 else 0.27)  # Shifted left
            y_pos = 1 - row_height * 0.5
            ax.text(x_pos, y_pos, col_name, fontsize=font_size, weight='bold', 
                    fontfamily='monospace', transform=ax.transAxes)
        
        # Draw row lines
        ax.axhline(y=1 - row_height, xmin=0.02, xmax=0.98, color='black', linewidth=1)
        
        # Draw data rows
        for row_idx, row_data in enumerate(rows):
            y_pos = 1 - (row_idx + 1.5) * row_height
            
            # IPC Group column
            ax.text(0.02, y_pos, row_data[0], fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Count column
            ax.text(0.17, y_pos, str(row_data[1]), fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Title column - positioned with no extra space
            ax.text(0.27, y_pos, row_data[2], fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Row separator line
            ax.axhline(y=1 - (row_idx + 2) * row_height, xmin=0.02, xmax=0.98, 
                       color='black', linewidth=0.5, alpha=0.3)
        
        # Draw vertical separator lines (moved left, removed right border)
        ax.axvline(x=0.02, ymin=0, ymax=1, color='black', linewidth=1)
        ax.axvline(x=0.17, ymin=0, ymax=1, color='black', linewidth=1)
        ax.axvline(x=0.27, ymin=0, ymax=1, color='black', linewidth=1)
        return plt


    def get_top_non_inventor_applicants(self, top_n=20):
        """
        Retrieves the most frequent applicants that are not also inventors.
        
        This function:
        1. Extracts applicants from the 'Applicants' column (separated by semicolons)
        2. Filters out applicants with non-Latin characters
        3. Compares them with inventors from the 'Inventors' column
        4. Creates a new dataframe for applicants that are not inventors with their IPC codes
        5. Groups similar applicant names (75% similarity) and counts them
        6. Returns a bar plot of the top N most frequent non-inventor applicants
        
        Args:
            top_n: Number of top applicants to display (default: 20)
            
        Returns:
            A matplotlib figure showing top non-inventor applicants
        """
        
        from collections import defaultdict
    
        # Create a list to hold all non-inventor applicants and their IPC codes
        non_inventor_applicants_data = []
        
        # Regular expression to check if string contains only Latin characters, numbers, and common punctuation
        latin_char_pattern = re.compile(r'^[A-Za-z0-9\s\.,;:\-\'\"\(\)&]+$')
        
        # Process each row in the filtered data
        for _, row in self.filtered_data.iterrows():
            # Get applicants and inventors
            applicants = str(row['Applicants']) if not pd.isna(row['Applicants']) else ""
            inventors = str(row['Inventors']) if not pd.isna(row['Inventors']) else ""
            ipc_codes = str(row['I P C']) if not pd.isna(row['I P C']) else ""
            
            # Split by semicolons and strip whitespace
            applicant_list = [app.strip() for app in applicants.split(';') if app.strip()]
            inventor_list = [inv.strip() for inv in inventors.split(';') if inv.strip()]
            
            # Filter out applicants with non-Latin characters
            applicant_list = [app for app in applicant_list if latin_char_pattern.match(app)]
            
            # Find applicants who are not inventors
            for applicant in applicant_list:
                # Check if this applicant name appears in the inventors list
                is_inventor = False
                for inventor in inventor_list:
                    # Use fuzzy matching to check if applicant name matches any inventor name
                    similarity = fuzz.ratio(applicant.lower(), inventor.lower())
                    if similarity > 85:  # High threshold for considering them the same
                        is_inventor = True
                        break
                
                # If applicant is not an inventor, add to our data
                if not is_inventor:
                    non_inventor_applicants_data.append({
                        'Applicant': applicant,
                        'IPC': ipc_codes
                    })
        
        # Create a dataframe from the collected data
        self.non_inventor_applicants_df = pd.DataFrame(non_inventor_applicants_data)
        
        # Now group similar applicant names using fuzzy matching
        # First, get unique applicant names
        unique_applicants = list(self.non_inventor_applicants_df['Applicant'].unique())
        
        # Create a mapping dictionary for grouping similar names
        name_groups = defaultdict(list)
        grouped_names = set()  # Track which names have been grouped
        
        # Create groups of similar names
        for i, name1 in enumerate(unique_applicants):
            if name1 in grouped_names:
                continue  # Skip if already in a group
                
            # Create a new group with this name
            group_key = name1
            name_groups[group_key].append(name1)
            grouped_names.add(name1)
            
            # Find similar names to add to this group
            for name2 in unique_applicants[i+1:]:
                if name2 in grouped_names:
                    continue  # Skip if already in a group
                    
                # Calculate similarity
                similarity = fuzz.ratio(name1.lower(), name2.lower())
                if similarity >= 75:  # 75% similarity threshold
                    name_groups[group_key].append(name2)
                    grouped_names.add(name2)
        
        # Count occurrences of each applicant group
        group_counts = {}
        for group_name, similar_names in name_groups.items():
            # Count the total occurrences of all names in this group
            count = sum(len(self.non_inventor_applicants_df[self.non_inventor_applicants_df['Applicant'] == name]) 
                        for name in similar_names)
            group_counts[group_name] = count
        
        # Get the top N applicant groups by count
        self.top_applicants = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # Split names and counts
        names, counts = zip(*self.top_applicants)
        
        # Reverse for horizontal bar chart (most frequent at top)
        names = names[::-1]
        counts = counts[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(np.arange(len(names)), counts, color='skyblue')
        
        # Add count labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                     str(counts[i]), va='center')
        
        # Truncate long names for display (optional)
        display_names = [name if len(name) <= 40 else name[:37] + '...' for name in names]
        
        # Format plot
        plt.yticks(np.arange(len(display_names)), display_names)
        plt.xlabel('Number of Applications')
        plt.title(f'Top {top_n} Most Frequent Non-Inventor Applicants')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Return the figure
        return plt
        

    def plot_applicant_parallel_coordinates(self, top_n=10, year_range=None):
        """Create parallel coordinates plot for top non-inventor applicants showing absolute publication counts by year."""
        if year_range is None:
            current_year = datetime.now().year
            year_range = (current_year - 20, current_year)     
            
        # Get top applicants if not already calculated
        if not hasattr(self, 'top_applicants'):
            self.get_top_non_inventor_applicants(top_n=top_n)
        
        sorted_applicants = sorted(self.top_applicants, key=lambda x: x[1], reverse=True)[:top_n]
        top_applicants = [name for name, _ in sorted_applicants]
        
        # Filter publication years
        all_years = sorted(self.filtered_data['PubYear'].dropna().unique().astype(int))
        
        if year_range:
            start_year, end_year = year_range
            all_years = [year for year in all_years if start_year <= year <= end_year]
            if not all_years:
                raise ValueError(f"No data available in the specified year range: {year_range}")
    
        # Create dataframe to hold counts
        applicant_yearly_counts = pd.DataFrame(index=top_applicants, columns=all_years, data=0)
        
        # Count publications per applicant and year
        for applicant in top_applicants:
            applicant_patents = self.filtered_data[
                self.filtered_data['Applicants'].str.contains(applicant, case=False, na=False)
            ]
            # Apply year range filter
            if year_range:
                applicant_patents = applicant_patents[
                    applicant_patents['PubYear'].between(start_year, end_year)
                ]
            
            yearly_counts = applicant_patents['PubYear'].value_counts().astype(int)
            for year, count in yearly_counts.items():
                if year in applicant_yearly_counts.columns:
                    applicant_yearly_counts.at[applicant, year] = count
        
        # Plot
        fig, host = plt.subplots(figsize=(15, 10))
        colors = plt.cm.tab10.colors
        
        for i, applicant in enumerate(top_applicants):
            data = applicant_yearly_counts.loc[applicant].values
            x = np.linspace(0, len(all_years) - 1, len(all_years))
            
            verts = [(x[0], data[0])]
            codes = [Path.MOVETO]
            for j in range(1, len(x)):
                x0, y0 = x[j - 1], data[j - 1]
                x1, y1 = x[j], data[j]
                ctrl1 = (x0 + (x1 - x0) / 3, y0)
                ctrl2 = (x0 + 2 * (x1 - x0) / 3, y1)
                verts.extend([ctrl1, ctrl2, (x1, y1)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=3, edgecolor=colors[i % len(colors)], alpha=0.8)
            host.add_patch(patch)



    def plot_applicant_ipc_bubble_chart(self, top_n=20):
        """
        Plots a bubble chart of the top applicants and their IPC groups using fuzzy name matching.
    
        Args:
            top_n: Number of top applicants to display (default: 20)
        """
        # Get the top applicants
        top_applicants = self.top_applicants
        top_names = [name for name, _ in top_applicants]
    
        # Create a dictionary to store IPC counts for each applicant
        applicant_ipc_counts = {applicant: Counter() for applicant in top_names}
    
        # Count IPC groups for each applicant (with fuzzy matching)
        for _, row in self.filtered_data.iterrows():
            applicants = str(row['Applicants']) if not pd.isna(row['Applicants']) else ""
            ipc_codes = str(row['I P C']) if not pd.isna(row['I P C']) else ""
    
            applicant_list = [app.strip() for app in applicants.split(';') if app.strip()]
            ipc_list = [ipc.strip()[:4] for ipc in ipc_codes.split(';') if ipc.strip()]
    
            for applicant in applicant_list:
                matched_applicant = None
                max_score = 0
                for known_applicant in top_names:
                    score = fuzz.token_sort_ratio(applicant, known_applicant)
                    if score > 75 and score > max_score:
                        matched_applicant = known_applicant
                        max_score = score
    
                if matched_applicant:
                    applicant_ipc_counts[matched_applicant].update(ipc_list)
    
        # Get the top 20 IPC groups
        all_ipcs = []
        for ipcs in self.filtered_data['I P C'].dropna():
            codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipcs))
            all_ipcs.extend([code[:4] for code in codes])
        top_ipcs = pd.Series(all_ipcs).value_counts().head(20).index.tolist()
    
        # Create the bubble chart
        plt.figure(figsize=(14, 10))
        for applicant, ipc_counts in applicant_ipc_counts.items():
            for ipc, count in ipc_counts.items():
                if ipc in top_ipcs:
                    size = (count + 1) * 50  
                    max_count = max(ipc_counts.values())
                    if max_count > 1:
                        color = plt.cm.coolwarm((count - 1) / (max_count - 1))
                    else:
                        color = 'blue'
                    plt.scatter(ipc, applicant, s=size, color=color, alpha=0.8, edgecolors='w')
    
        plt.xlabel('IPC Groups')
        plt.ylabel('Applicants')
        plt.title('Bubble Chart of Top Applicants and IPC Groups (bubble color is warmer and size bigger when higher count)')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return plt




    def generate_wordclouds_by_pos(self, text_column='Abstract', pospeech='Nouns'):
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'Lato-Regular.ttf')
        stopwords = self.load_stopwords()

        text_data = self.filtered_data[text_column].dropna().astype(str).str.cat(sep=' ')
        tokens = word_tokenize(text_data.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords]

        tagged_tokens = pos_tag(tokens)
        group_words = [word for word, tag in tagged_tokens if tag in pos_groups[pospeech]]
        word_freq = Counter(group_words)

        wordcloud = WordCloud(
            font_path=font_path,
            width=800, height=400,
            background_color='white',
            colormap='Dark2'
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{pospeech} Word Cloud', fontsize=16)
        ax.axis('off')
        fig.tight_layout()

        return fig   # not plt

    def plot_parallel_coordinates(self, top_n=5, year_range=None):
        if year_range is None:
            current_year = datetime.now().year
            year_range = range(current_year - 20, current_year)     

        df_filtered = self.filtered_data.copy()
        df_filtered = df_filtered[df_filtered['MainIPC'].notna()]
        df_filtered['PubYear'] = pd.to_numeric(df_filtered['PubYear'], errors='coerce').fillna(0).astype(int)
        df_filtered = df_filtered[df_filtered['PubYear'] > 0]

        ipc_counts = df_filtered['MainIPC'].value_counts()
        top_ipcs = ipc_counts.head(top_n).index

        ipc_yearly_counts = pd.DataFrame(index=top_ipcs, columns=year_range, data=0)
        for ipc in top_ipcs:
            ipc_data = df_filtered[df_filtered['MainIPC'] == ipc]
            yearly_counts = ipc_data['PubYear'].value_counts()
            for year in year_range:
                ipc_yearly_counts.at[ipc, year] = yearly_counts.get(year, 0)

        normalized_data = ipc_yearly_counts.div(ipc_yearly_counts.max(axis=1), axis=0)

        fig, host = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10.colors

        for i, ipc in enumerate(top_ipcs):
            data = normalized_data.loc[ipc].values
            x = np.linspace(0, len(year_range) - 1, len(year_range))

            verts = [(x[0], data[0])]
            codes = [Path.MOVETO]
            for j in range(1, len(x)):
                x0, y0 = x[j - 1], data[j - 1]
                x1, y1 = x[j], data[j]
                ctrl1 = (x0 + (x1 - x0) / 3, y0)
                ctrl2 = (x0 + 2 * (x1 - x0) / 3, y1)
                verts.extend([ctrl1, ctrl2, (x1, y1)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=4, edgecolor=colors[i % len(colors)])
            host.add_patch(patch)

        host.set_xlim(0, len(year_range) - 1)
        host.set_xticks(range(len(year_range)))
        host.set_xticklabels(year_range, fontsize=10, rotation=45)
        host.set_title(
            f'Parallel Coordinates Plot: Top {top_n} IPC Codes ({year_range[0]}–{year_range[-1]})',
            fontsize=16
        )
        host.set_xlabel('Publication Year')
        host.set_ylabel('Normalized Patent Counts')
        plt.legend(top_ipcs, title='Main IPC', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt

    def plot_applicant_parallel_coordinates(self, top_n=10, year_range=None):
        """Create parallel coordinates plot for top non-inventor applicants showing normalized publication counts by year."""
        if year_range is None:
            current_year = datetime.now().year
            year_range = (current_year - 20, current_year)
    
        start_year, end_year = year_range
    
        # Ensure 'PubYear' is in integer format
        self.filtered_data['PubYear'] = pd.to_numeric(self.filtered_data['PubYear'], errors='coerce').dropna().astype(int)
    
        # Get top applicants if not already calculated
        if not hasattr(self, 'top_applicants'):
            self.get_top_non_inventor_applicants(top_n=top_n)
    
        # Sort applicants by count and take top_n
        sorted_applicants = sorted(self.top_applicants, key=lambda x: x[1], reverse=True)[:top_n]
        top_applicants = [name for name, _ in sorted_applicants]
    
        # Filter years
        all_years = sorted([year for year in self.filtered_data['PubYear'].unique() if start_year <= year <= end_year])
        if not all_years:
            raise ValueError("No valid publication years in the selected range")
    
        # Initialize dataframe with zeros
        applicant_yearly_counts = pd.DataFrame(index=top_applicants, columns=all_years, data=0)
    
        # Count publications for each applicant by year using fuzzy match (≥75%)
        for applicant in top_applicants:
            matching_rows = self.filtered_data[
                self.filtered_data['Applicants'].fillna('').apply(
                    lambda x: fuzz.partial_ratio(x.lower(), applicant.lower()) >= 75
                )
            ]
            yearly_counts = matching_rows['PubYear'].value_counts().astype(int)
            for year, count in yearly_counts.items():
                if year in applicant_yearly_counts.columns:
                    applicant_yearly_counts.at[applicant, year] = count
    
        # Normalize counts per applicant between 0 and 1
        normalized_counts = applicant_yearly_counts.div(applicant_yearly_counts.max(axis=1), axis=0).fillna(0)
    
        # Plotting
        fig, host = plt.subplots(figsize=(15, 10))
        colors = plt.cm.tab10.colors
    
        for i, applicant in enumerate(top_applicants):
            data = normalized_counts.loc[applicant].values.astype(float)
            x = np.linspace(0, len(all_years) - 1, len(all_years))
    
            # Bezier curve
            verts = [(x[0], data[0])]
            codes = [Path.MOVETO]
            for j in range(1, len(x)):
                x0, y0 = x[j - 1], data[j - 1]
                x1, y1 = x[j], data[j]
                ctrl1 = (x0 + (x1 - x0) / 3, y0)
                ctrl2 = (x0 + 2 * (x1 - x0) / 3, y1)
                verts.extend([ctrl1, ctrl2, (x1, y1)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
    
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=3, edgecolor=colors[i % len(colors)], alpha=0.8)
            host.add_patch(patch)
    
        # Axis and labels
        host.set_xlim(0, len(all_years) - 1)
        host.set_ylim(0, 1.05)
        host.set_xticks(range(len(all_years)))
        host.set_xticklabels(all_years, fontsize=10, rotation=45)
        host.set_title(f'Top {top_n} Non-Inventor Applicants by Normalized Publication Count', fontsize=16, pad=20)
        host.set_xlabel('Publication Year', fontsize=12)
        host.set_ylabel('Normalized Publication Count (0–1)', fontsize=12)
        host.grid(True, linestyle='--', alpha=0.6)
    
        # Legend
        plt.subplots_adjust(right=0.7)
        legend_entries = []
        for i, (applicant, count) in enumerate(sorted_applicants):
            display_name = applicant if len(applicant) <= 30 else applicant[:27] + '...'
            legend_entries.append(f"{display_name} (Total: {count})")
    
        legend_patches = [
            patches.Patch(color=colors[i % len(colors)], label=legend_entries[i])
            for i in range(len(legend_entries))
        ]
    
        host.legend(handles=legend_patches,
                    title='Applicants with Total Publications',
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    borderaxespad=0.,
                    fontsize=10)
    
        plt.tight_layout()
        return plt

    '''
    Patent Network section (adapted to return both figure + dataframe)
    '''

    def _clean_applicant_name(self, name):
        """Clean applicant names for network analysis"""
        name = name.strip()
        if name.lower().endswith(', inc') or name.lower().endswith(', inc.'):
            name = name[:name.lower().rfind(', inc')].strip()
        return name

    def _has_similar_value(self, applicant, inventors):
        """Check if applicant name is similar to any inventor name"""
        applicant = applicant.lower().strip()
        for inventor in inventors:
            inventor = inventor.lower().strip()
            if any(applicant[i:i+5] in inventor for i in range(len(applicant) - 4)):
                return True
        return False

    def _deduplicate_applicants(self, applicants, threshold=80):
        """
        Merge similar applicant names using fuzzy matching (>= threshold).
        Keeps the first seen version as the canonical name.
        """
        canonical_names = []
        mapping = {}

        for name in applicants:
            if not canonical_names:
                canonical_names.append(name)
                mapping[name.lower()] = name
                continue

            # Compare to existing canonical names
            match, score, _ = process.extractOne(
                name, canonical_names, scorer=fuzz.token_sort_ratio
            )
            if score >= threshold:
                # Map to the existing canonical name
                mapping[name.lower()] = match
            else:
                # New canonical name
                canonical_names.append(name)
                mapping[name.lower()] = name

        return mapping

    def filter_data_for_network(self):
        """Filter and prepare data for network analysis"""
        data_copy = self.data.copy()
        data_copy['Applicants'] = data_copy['Applicants'].fillna('').str.split(';')
        data_copy['Inventors'] = data_copy['Inventors'].fillna('').str.split(';')

        filtered_rows = []
        for _, row in data_copy.iterrows():
            applicants = row['Applicants']
            inventors = row['Inventors']

            if any(applicant.strip().lower() == inventor.strip().lower()
                   for applicant in applicants for inventor in inventors):
                continue

            filtered_applicants = [
                self._clean_applicant_name(applicant)
                for applicant in applicants
                if not self._has_similar_value(applicant, inventors)
            ]

            selected_applicant = filtered_applicants[0] if filtered_applicants else None
            if selected_applicant:
                filtered_rows.append({
                    "Applicant": selected_applicant,
                    "Inventors": '; '.join(inventors)
                })

        filtered_df = pd.DataFrame(filtered_rows)
        
        # Compact applicants
        compacted_data = (
            filtered_df.groupby(filtered_df['Applicant'].str.lower())
            .agg({"Inventors": lambda x: '; '.join(set('; '.join(x).split('; ')))})
            .reset_index()
        )
        compacted_data.columns = ['Applicant_Lower', 'Inventors']
        # --- NEW: Deduplicate similar applicant names ---
        unique_applicants = compacted_data['Applicant_Lower'].tolist()
        dedup_map = self._deduplicate_applicants(unique_applicants, threshold=80)

        compacted_data['Applicant_Canonical'] = compacted_data['Applicant_Lower'].map(dedup_map)

        # Group again after deduplication
        compacted_data = (
            compacted_data.groupby('Applicant_Canonical')
            .agg({"Inventors": lambda x: '; '.join(set('; '.join(x).split('; ')))})
            .reset_index()
        )
        compacted_data.columns = ['Applicant_Lower', 'Inventors']

        # Keep mapping to original names
        applicant_mapping = {}
        for _, row in filtered_df.iterrows():
            lower_name = row['Applicant'].lower()
            if lower_name not in applicant_mapping:
                applicant_mapping[lower_name] = row['Applicant']

        compacted_data['Applicant_Original'] = compacted_data['Applicant_Lower'].map(
            lambda x: applicant_mapping.get(x, x)
        )
        self.network_filtered_data = compacted_data
        self.applicant_mapping = applicant_mapping
        return compacted_data

    def build_network_graph(self):
        """Build network graph from filtered data"""
        if not hasattr(self, 'network_filtered_data') or self.network_filtered_data is None:
            raise ValueError("Data must be filtered first using filter_data_for_network()")

        G = nx.Graph()
        for i, row1 in self.network_filtered_data.iterrows():
            for j, row2 in self.network_filtered_data.iterrows():
                if i >= j:
                    continue
                inventors1 = set([inv.strip() for inv in row1['Inventors'].split('; ') if inv.strip()])
                inventors2 = set([inv.strip() for inv in row2['Inventors'].split('; ') if inv.strip()])
                shared_inventors = inventors1 & inventors2
                weight = len(shared_inventors)
                if weight > 0:
                    G.add_edge(
                        row1['Applicant_Lower'], 
                        row2['Applicant_Lower'], 
                        weight=weight, 
                        shared_inventors=shared_inventors
                    )

        self.network_graph = G
        return G

def _contains_non_latin(self, text):
        """Check if text contains non-Latin characters (CJK, Arabic, etc.)"""
        for char in str(text):
            if ord(char) > 0x024F:  # Beyond extended Latin range
                return True
        return False

    def _translate_labels(self, labels_dict):
        """
        Translate any non-Latin labels to English using OpenAI.
        labels_dict: {node_key: display_label}
        Returns updated dict with translations.
        """
        import openai
        import os
        import json

        # Find labels that need translation
        to_translate = {
            node: label 
            for node, label in labels_dict.items() 
            if self._contains_non_latin(label)
        }

        if not to_translate:
            print("No non-Latin labels found, skipping translation.")
            return labels_dict

        print(f"\nTranslating {len(to_translate)} non-Latin label(s):")
        for node, label in to_translate.items():
            print(f"  '{label}'")

        # Build a single batch request to minimise API calls
        names_list = list(to_translate.values())
        prompt = (
            "Translate the following organization/company names to English. "
            "Keep them concise. Return ONLY a JSON array of translated strings "
            "in the same order as the input. No explanation, no extra text.\n\n"
            "Input:\n" + json.dumps(names_list, ensure_ascii=False)
        )

        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # cheap and fast
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            translated_list = json.loads(raw)

            if len(translated_list) != len(names_list):
                raise ValueError(
                    f"Expected {len(names_list)} translations, got {len(translated_list)}"
                )

            # Build updated labels dict
            updated_labels = dict(labels_dict)  # copy
            for (node, original), translation in zip(to_translate.items(), translated_list):
                print(f"  '{original}' -> '{translation}'")
                updated_labels[node] = translation

            return updated_labels

        except Exception as e:
            print(f"Warning: Translation failed ({e}). Using original labels.")
            return labels_dict  # fall back gracefully

    def generate_network_image(self, top_n=20):
        """Generate network visualization and also return DataFrame with connections"""
        if not hasattr(self, 'network_graph') or self.network_graph is None:
            raise ValueError("Graph must be built first using build_network_graph()")

        # Pick top N nodes
        degrees = sorted(self.network_graph.degree, key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes = [node for node, _ in degrees]
        H = self.network_graph.subgraph(top_nodes)

        # Create matplotlib Figure
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(12, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        pos = nx.circular_layout(H)

        # Build raw display labels from applicant mapping
        raw_labels = {
            node: self.applicant_mapping.get(node, node) 
            for node in H.nodes()
        }

        # ✅ Translate any non-Latin labels to English
        display_labels = self._translate_labels(raw_labels)

        nx.draw(
            H, pos, ax=ax, labels=display_labels, with_labels=True,
            node_size=3000, font_size=8, font_weight='bold',
            edge_color='gray', node_color='skyblue',
        )
        labels = nx.get_edge_attributes(H, 'weight')
        nx.draw_networkx_edge_labels(
            H, pos, edge_labels=labels, font_size=8, ax=ax
        )
        ax.set_title(f"Top {top_n} Most Connected Applicants", fontsize=16)

        # Build connections DataFrame
        results = []
        for applicant_lower in top_nodes:
            neighbors = list(self.network_graph.neighbors(applicant_lower))
            applicant_original = self.applicant_mapping.get(applicant_lower, applicant_lower)
            for neighbor_lower in neighbors:
                edge_data = self.network_graph[applicant_lower][neighbor_lower]
                shared_inventors = edge_data.get('shared_inventors', set())
                neighbor_original = self.applicant_mapping.get(neighbor_lower, neighbor_lower)
                if shared_inventors:
                    results.append({
                        "Most Connected Applicant": applicant_original,
                        "Connected To": neighbor_original,
                        "Shared Inventors": "; ".join(sorted(shared_inventors))
                    })

        connections_df = pd.DataFrame(
            results, 
            columns=["Most Connected Applicant", "Connected To", "Shared Inventors"]
        )

        # Deduplicate A-B vs B-A
        if not connections_df.empty:
            connections_df[['Applicant1', 'Applicant2']] = pd.DataFrame(
                connections_df[['Most Connected Applicant', 'Connected To']].apply(
                    lambda x: sorted([x.iloc[0], x.iloc[1]]), axis=1
                ).tolist()
            )
            connections_df = (
                connections_df.drop_duplicates(
                    subset=['Applicant1', 'Applicant2', 'Shared Inventors']
                )
                .drop(columns=['Applicant1', 'Applicant2'])
            )

        # Sort by number of shared inventors then alphabetically
        if not connections_df.empty:
            connections_df['Num_Shared'] = connections_df['Shared Inventors'].str.count(';') + 1
            connections_df = (
                connections_df.sort_values(
                    ['Num_Shared', 'Most Connected Applicant'], 
                    ascending=[False, True]
                )
                .drop(columns=['Num_Shared'])
                .reset_index(drop=True)
            )

        self.connections_db = connections_df
        return fig

    def get_connections_df(self):
        """
        Return the stored DataFrame of applicant connections.
        """
        if hasattr(self, "connections_db") and self.connections_db is not None:
            return self.connections_db
        else:
            raise ValueError("No connections DataFrame available. Run generate_network_image() first.")



    """

                                                               _
                                                              _)\.-.
                                             .-.__,___,_.-=-. )\`  a`\_
                                         .-.__\__,__,__.-=-. `/  \     `\
                                         {~,-~-,-~.-~,-,;;;;\ |   '--;`)/
                                          \-,~_-~_-,~-,(_(_(;\/   ,;/
                                           ",-.~_,-~,-~,)_)_)'.  ;;(                          O    P    E    N
                                             `~-,_-~,-~(_(_(_(_\  `;\
                                       ,          `"~~--,)_)_)_)\_   \                             A .  I .
                                       |\              (_(_/_(_,   \  ;
                                       \ '-.       _.--'  /_/_/_)   | |                   A  N  A  L  Y  T  I  C   S
                                        '--.\    .'          /_/    | |
                                            ))  /       \      |   /.'
                                           //  /,        | __.'|  ||
                                          //   ||        /`    (  ||
                                         ||    ||      .'       \ \\
                                         ||    ||    .'_         \ \\
                                          \\   //   / _ `\        \ \\__
                                    wdj   \'-'/(   _  `\,;        \ '--:,
                                            `"`  `"` `-,,;         `"`",,;


    """

    def get_top_ipcs_AI_defined(self, title_file="cpc_group_titles.csv"):
        """
        Uses stored top IPC counts and enriches them with titles from the CPC definition CSV.
        Then filter the dataframe by each of the top IPC groups, retrieve a maximum of 20 documents per
        IPC group and use OpenAI to summarize their contents.
        The summary is then added to the IPC group title as a more related description 
        """
        if not hasattr(self, 'top_ipc_counts'):
            raise ValueError("Top IPCs not calculated yet. Please call plot_top_ipcs() first.")
        
        # --- Load IPC titles ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        title_path = os.path.join(current_dir, title_file)
        if not os.path.isfile(title_path):
            raise FileNotFoundError(f"Title file '{title_file}' not found at {title_path}.")
        df_titles = pd.read_csv(title_path)
        title_dict = dict(zip(df_titles['Group'], df_titles['Title']))

        # --- OpenAI client ---
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        model = "gpt-3.5-turbo"
        enc = tiktoken.encoding_for_model(model)
        chunk_token_limit = 3000


        def num_tokens(text):
            return len(enc.encode(text))


        def summarize_chunk(chunk_text, ipc_code):
            system_msg = "You are a patent analyst. Summarize the common technical features and problems addressed by these patents."
            user_msg = f"Here is a set of patent abstracts classified under IPC {ipc_code}:\n\n{chunk_text}\n\nWhat do they have in common?"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()

        # --- Build results with AI summaries ---
        results = []
        for ipc in self.top_ipc_counts.index[:6]:
            count = int(self.top_ipc_counts[ipc])
            title = title_dict.get(ipc, "No title found")

            # Get up to 20 abstracts for this IPC group
            subset = self.filtered_data[self.filtered_data['MainIPC'] == ipc]
            abstracts = subset['Abstract'].dropna().tolist()
            if len(abstracts) > 20:
                abstracts = subset.sample(n=20, random_state=42)['Abstract'].dropna().tolist()

            if not abstracts:
                ai_summary = "No abstracts available for summarization."
            else:
                # Chunk abstracts into token-limited groups
                chunks, current_chunk, current_tokens = [], [], 0
                for abstract in abstracts:
                    tokens = num_tokens(abstract)
                    if current_tokens + tokens > chunk_token_limit:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk, current_tokens = [abstract], tokens
                    else:
                        current_chunk.append(abstract)
                        current_tokens += tokens
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Summarize each chunk
                summaries = [summarize_chunk(chunk, ipc) for chunk in chunks]

                # Final summary
                final_prompt = (
                    "Here are several summaries of patent groups. "
                    "Provide a concise, no more than 80 words, overall summary of what these patents have in common. "
                    "Start the summary with the sentence, The patents in this group:\n\n"
                    + "\n\n".join(summaries)
                )
                final_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a technical summarizer for patent portfolios."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.3
                )
                ai_summary = final_response.choices[0].message.content.strip()

            results.append([ipc, count, title, ai_summary])

        html_table = self.ipc_ai_results_to_html_table(results)
        return html_table
    
    def get_top_applicants_AI_description(self, applicants, top_ipc_n=5, title_file="cpc_group_titles.csv"):
        """
        Generates a professional CEO-style competitor intelligence report 
        for the given applicants, based on top IPCs.
        """

        # --- Step 1: Get top IPCs with titles ---
        top_ipcs = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        title_path = os.path.join(current_dir, title_file)

        if not os.path.isfile(title_path):
            raise FileNotFoundError(f"Title file '{title_file}' not found at {title_path}.")
        
        df_titles = pd.read_csv(title_path)
        title_dict = dict(zip(df_titles['Group'], df_titles['Title']))

        for ipc in self.top_ipc_counts.index[:top_ipc_n]:
            title = title_dict.get(ipc, "No title found")
            top_ipcs.append(f"{ipc} {title}")

        top_ipcs_text = "; ".join(top_ipcs)

        # --- Step 2: Applicants list ---
        applicants_text = "[" + ", ".join([f"'{a}'" for a in applicants]) + "]"

        # --- Step 3: Prompt ---
        system_msg = "You are a patent attorney. Prepare professional competitor intelligence reports."
        user_msg = f"""
    I have made a patent search based on the IPC groups {top_ipcs_text}.
    There are several companies that appeared: {applicants_text}.
    If you don't find an exact match try to find a close approximation.

    Prepare a professional report for the CEO of a company. 
    The report must be a table with a header containing the fields:

    Company | Match Status (Exact / Approximate) | What They Do / Their Activities Related to the relevant IPCs | IPC Relevance
    """

        # --- Step 4: OpenAI API call ---
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    """
                                             _ _ _
                                            '.` `.'
                                            _ >_< _
                                       _.-'`-.-=-.-`'-._
                                     .;'.--./     \.--.';.
                                   .'  >    \     /    <  '.
                                  /.__./\__.';-=-;'.__/\.__.\
                                  |  \ /   \/     \/   \ /  |
                                  ;   |    |       |    |   |
                                   '._/\   /\     /\   /\_.'
                                        `-`  ;-.-;2 `-` 
                                              > <
                                          .-'{___}'-.
                                         '-.._____..-'
                                              |||
                                              |||
                                             /'-'\
                                          .-',_ _,'-.
                                         /'-. .`. .-'\
                                      .='  . `. .` .  '=.
                                       '-'`-./   \.-`'-'
                                             '._.'
    """

    def ai_markdown_to_html_table(self, md_text):
        lines = [line.strip() for line in md_text.split("\n") if line.strip()]
        if len(lines) < 3:
            return "<p>No AI data available</p>"

        header = lines[1].split('|')[1:-1]  # remove empty columns
        rows = [line.split('|')[1:-1] for line in lines[3:]]

        html = '<table class="table table-striped table-bordered">'
        html += '<thead><tr>' + ''.join(f'<th>{h.strip()}</th>' for h in header) + '</tr></thead>'
        html += '<tbody>'
        for row in rows:
            html += '<tr>' + ''.join(f'<td>{cell.strip()}</td>' for cell in row) + '</tr>'
        html += '</tbody></table>'
        return html

    def ipc_ai_results_to_html_table(self, results):
        """
        Convert top IPC results (IPC code, count, title, AI summary) into an HTML table for rendering.
        """
        html = ['<table class="table table-striped table-hover"><thead>'
                '<tr><th>IPC</th><th>Count</th><th>Title</th><th>Summary</th></tr></thead><tbody>']

        for ipc, count, title, summary in results:
            html.append(f"<tr>"
                        f"<td>{ipc}</td>"
                        f"<td>{count}</td>"
                        f"<td>{title}</td>"
                        f"<td>{summary}</td>"
                        f"</tr>")

        html.append("</tbody></table>")
        return "\n".join(html)


    def connections_to_html_table(self, connections_df):
        """
        Convert a connections DataFrame into an HTML table.
        Expects columns: Most Connected Applicant, Connected To, Shared Inventors
        """
        if connections_df is None or connections_df.empty:
            return "<p>No applicant connections found.</p>"

        html = ['<table class="table table-striped table-hover"><thead>'
                '<tr><th>Most Connected Applicant</th>'
                '<th>Connected To</th>'
                '<th>Shared Inventors</th></tr></thead><tbody>']

        for _, row in connections_df.iterrows():
            html.append(
                f"<tr>"
                f"<td>{row['Most Connected Applicant']}</td>"
                f"<td>{row['Connected To']}</td>"
                f"<td>{row['Shared Inventors']}</td>"
                f"</tr>"
            )

        html.append("</tbody></table>")
        return "\n".join(html)
