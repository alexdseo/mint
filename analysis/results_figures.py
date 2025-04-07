import numpy as np
import pandas as pd
from analysis.utils import *
from scipy.stats import pearsonr, ttest_ind, levene
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ast
import argparse


class GetResults:
    """
        Get results and figures in the manuscript.
    """
    def __init__(self,
                df,
                random_seed = 2025,
                psig = 0.05,
                pop_threshold = 100000
        ):
        self.random_seed = random_seed
        self.df = rename(df) # Rename columns
        self.psig = psig
        self.pop_threshold = pop_threshold
        self.bar_width = 0.2
        self.bar_colors = ['#4878D0', '#D65F5F', '#6ACC64','#EE854A']
        self.features = ['FEND', 'RND_STDEV', 'CDC mRFEI', 'USDA %LowAccess', 'LSR Density', '#Restaurants',
                        'Obesity Prevalence', 'Diabetes Prevalence', 'CHD Prevalence', 'Total Population (log-scaled)',
                        '%Black',  '%Asian', '%Hispanic', '%White', 'Median Age', 'Median Income', 'Employment Rate',
                        '%LowSkillJob', '%CollegeEdu', '%PublicTransportation', '%LongComute', 'Gini Index']
        self.covariates = ['%Black',  '%Asian', '%Hispanic', '%White', 'Median Age', 'Median Income', 'Employment Rate',
                           '%LowSkillJob', '%CollegeEdu', '%PublicTransportation', '%LongComute', 'Gini Index',
                           'Total Population (log-scaled)']
        
        
        
    def correlation_heatmap(self):
        # Correlation heatmap including food environment metrics, diet-related diseases, SES factors
        pvals = self.df[self.features].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*self.df[self.features].corr().shape)
        # Set background color 
        sns.set_style(style = 'white')
        # Figure size # Significance threshold
        f, ax = plt.subplots(figsize=(15, 15))
        # Add diverging colormap from red to blue # Significant correlation
        sns.heatmap(self.df[self.features].corr()[pvals<self.psig], square=True, annot=False, cmap="coolwarm",
                    vmin=-1, fmt='.2f', annot_kws={'size': 17.5}, linewidth=.5, cbar_kws={"shrink": .75}, ax=ax)
        # Insignificant correlation
        sns.heatmap(self.df[self.features].corr()[pvals>=self.psig], square=True, annot=False, cbar=False,
                    cmap=sns.color_palette("binary", n_colors=1, desat=1), annot_kws={'size': 17.5})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
        # Show & Save plot
        # plt.show()
        f.savefig("corr_heatmap.pdf", bbox_inches='tight')
    
    def summary_stats(self, df_all, output_file="summary_stats.md"):
        # Rename columns
        renamed_df = rename(df_all)
        # Print out summary statistics
        summary_stats = renamed_df[self.features].describe(percentiles=[0.25, 0.5, 0.75]).T
        summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        # Renaming columns
        summary_stats.columns = ['Mean', 'STDEV', 'Min', 'First Quartile', 'Median', 'Third Quartile', 'Max']

        # Export to markdown file
        with open(output_file, "w") as f:
            f.write(summary_stats.to_markdown())

    def metro_rural_test(self):
        # Statistical test performed on comparing FEND score in metropolitan and rural county
        # Metropolitan defined with >100k in population
        metro = self.df[self.df['total_pop'] > self.pop_threshold]['FEND']   # 594 county
        rural = self.df[self.df['total_pop'] <= self.pop_threshold]['FEND']  # 608 county
        # Check variance equality
        _, p_levene = levene(metro, rural)
        equal_var = p_levene > 0.05  # If p > 0.05, assume equal variances
        if p_levene < 0.05:
            print("Reject null hypothesis: Variances between the groups are significantly different.")
        else:
            print("Fail to reject null hypothesis: Variances between the groups are not significantly different.")
        # Perform independent t-test (Welch’s test if variances are unequal)
        t_statistic, p_value = ttest_ind(metro, rural, equal_var=equal_var)
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)
        # Interpret results
        if p_value < 0.05:
            print("Reject null hypothesis: Mean between the groups are significantly different.")
        else:
            print("Fail to reject null hypothesis: Mean between the groups are not significantly different.")

    @staticmethod
    def decile_plot(decile, fe_metric, df):
        title = {
            'decile_income': 'Median Income'
        }.get(decile, 'Obesity Prevalence')
        colors = {
            'LSR Density': '#FC6A03', # Orange
            'CDC mRFEI': '#ff073a' # Red
        }.get(fe_metric, '#32CD32') # Default: 'USDA %LowAccess' # Green
        markers = {
            'LSR Density': '^', # up arrow
            'CDC mRFEI': 'v' # down arrow
        }.get(fe_metric, 's') # Default: 'USDA %LowAccess' # square
        # Plot
        f, ax = plt.subplots(figsize=(8, 6))
        ax = sns.pointplot(x=decile, y='FEND', data=df, linestyle='none', markersize=8.5, capsize=0.5,
                           err_kws={'linewidth': 2}, errorbar=('se'), n_boot=1000, color='#000492', ax=ax)
        # Set x-tick positions and labels
        ax.set_xticks(range(0, 10))
        ax.set_xticklabels(['Least'] + [str(i) for i in range(2, 10)] + ['Most'], fontsize=23)
        ax.set_xlabel(f"{title} Deciles", fontsize=23)
        
        # Create a second y-axis for the overlapping decile plot
        ax2 = ax.twinx()
        # Plot the overlapping decile plot on the second y-axis
        ax2 = sns.pointplot(x=decile, y=fe_metric, data=df, linestyle='none', markersize=8.5, capsize=0.5,
                            err_kws={'linewidth': 2}, errorbar=('se'), n_boot=1000, marker=markers,
                            color=colors, ax=ax2)
        # Set the label for the second y-axis
        ax.set_ylabel('Mean FEND', color='#000492', fontsize=23)
        ax2.set_ylabel(f"Mean {fe_metric}", color=colors, fontsize=23)
        ax.set_yticklabels(ax.get_yticklabels(), color='#000492', fontsize=23)
        ax2.set_yticklabels(ax2.get_yticklabels(), color=colors, fontsize=23)

        f.savefig(f"{title}_{fe_metric}_decile.pdf", bbox_inches='tight')

    def decile_analysis(self):
        # Decile analysis
        # Get deciles for obesity and median income
        self.df['decile_obesity'] = pd.qcut(self.df['Obesity Prevalence'], 10, labels=range(1, 11))
        self.df['decile_income'] = pd.qcut(self.df['Median Income'], 10, labels=range(1, 11))
        # Plot 6 decile plots
        for deciles in ['decile_obesity', 'decile_income']:
            for metrics in ['USDA %LowAccess', 'LSR Density', 'CDC mRFEI']:
                self.decile_plot(decile=deciles, fe_metric=metrics, df=self.df)
    
    def ee_barplot(self, df):
        # Effect estimates barplot
        # Filter to include only Adjusted models
        df = df[df['Model'] == 'Adj']
        # Reorder the DataFrame 
        df["Label Order"] = df["Health outcomes"].apply(lambda x: ['CHD', 'Diabetes','Obesity'].index(x))
        df = df.sort_values(["Health outcomes", "Label Order"]).reset_index(drop=True)

        # Set background color and figure size
        sns.set_style("whitegrid", {'axes.grid': False, 'grid.linestyle': ':'})
        f, ax = plt.subplots(figsize=(32, 24))

        # Define the bar width and positions
        bar_positions = {
            "FEND": np.arange(len(df)),
            "CDC mRFEI": np.arange(len(df)) + 3 * self.bar_width,
            "USDA %Low Access": np.arange(len(df)) + self.bar_width,
            "LSR Density": np.arange(len(df)) + 2 * self.bar_width
        }

        # Plot bars for each variable
        for i, (variable, positions) in enumerate(bar_positions.items()):
            for idx, (pos, val, ci, sig) in enumerate(zip(positions, df[variable], df[f"{variable} (95% CI)"].apply(ast.literal_eval), df[f"is_significant ({variable})"])):
                alpha = 1.0 if sig else 0.3
                lower_error = val - ci[0]
                upper_error = ci[1] - val
                bar = ax.barh(pos, val, xerr=[[lower_error], [upper_error]], height=self.bar_width, label=variable if idx == 0 else "",
                            color=self.bar_colors[i], align='center', ecolor='black', capsize=15, error_kw=dict(lw=3.5, capthick=3.5), alpha=alpha)
                # Add outlines
                bar[0].set_edgecolor('black')
                bar[0].set_linewidth(3.5)

        # Set y-axis labels and ticks
        ax.set_yticks(np.arange(len(df)) + 0.35)
        ax.set_yticklabels(df['Health outcomes'], fontsize=70,fontweight='bold')
        # Set titles and labels
        ax.set_xlabel('Effect Estimates (β)', fontsize=70, fontweight='bold')
        ax.tick_params(axis='x', labelsize=65)
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=3.5)
        # Add background shading
        for i in range(0, len(df), 1):
            ax.axhspan(i-0.1, i+0.70, facecolor='gray', alpha=0.075)
        # Adjust layout
        plt.tight_layout()

        # Legend
        legend_handles = [
            Patch(facecolor=self.bar_colors[i], edgecolor='black', label=variable, linewidth=3.5)
            for i, variable in enumerate(bar_positions.keys())
        ]
        # Move the legend outside the plot area
        plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=65)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)

        # Save plot
        f.savefig("main_ee_barplot.pdf", bbox_inches='tight')

    def extra_lm(self, df_ct, ruca, th):
        # Additional linear model results using data other than county-level
        # Features relevant to census tract analysis # No LSR Density
        ct_features = [item for item in self.features if item != 'LSR Density']
        # RUCA scores to determine rural & metropolitan area for census tracts
        metro_ct, rural_ct = metro_rural_ct(df_ct, ruca)
        main_ct = thresholding(th, df_ct) # Considering census tracts with more than 5 restaurants
        # Normalize for linear modeling
        metro_ct, rural_ct = normalize(metro_ct, ct_features), normalize(rural_ct, ct_features)
        main_ct = normalize(main_ct, ct_features)
        # FEND/RND_STDEV ~ SES
        vif_vars = [item for item in self.covariates if item != '%White'] # Remove '%White' based on vif results
        for metric in ['FEND', 'RND_STDEV']:
            metro_model = linear_model(metro_ct, metric, vif_vars) # Use vif_vars for metro_model
            rural_model = linear_model(rural_ct, metric, self.covariates)
            main_model = linear_model(main_ct, metric, self.covariates)
            # Report
            lines = [f"# {metric}~SES Model Summary Report\n"]
            for name, model in zip(["Metro Model", "Rural Model", f"Census tract Model w/ {th}% thershold on #Restaurants"],
                                   [metro_model, rural_model, main_model]):
                lines.append(f"## {name}\n")
                lines.append(f"```\n{model.summary()}\n```\n")
            # File name
            filename = f"{metric}_SES_ct_report.md".replace(" ", "_") 
            with open(filename, "w") as f:
                f.writelines("\n".join(lines))

        # Diet-related disease ~ FE metrics + SES using main_ct
        for ho in ['Obesity Prevalence', 'Diabetes Prevalence', 'CHD Prevalence']:
            # Report
            lines = [f"# {ho}~FE metrics+SES Model Summary Report\n"]
            for metric in ['FEND', 'RND_STDEV', 'USDA %LowAccess', 'CDC mRFEI']:
                all_vars = [metric] + vif_vars
                if metric in ['USDA %LowAccess', 'CDC mRFEI']:
                    adjm_df = remove_na(main_ct, metric) # Remove NA data for 'USDA %LowAccess', 'CDC mRFEI' in census tracts
                else:
                    adjm_df = main_ct
                model = linear_model(adjm_df, ho, all_vars)
                # Write lines 
                lines.append(f"## Metric: {metric}\n")
                lines.append(f"```\n{model.summary()}\n```\n")
            # File name
            filename = f"{ho}_AdjModel_ct_report.md".replace(" ", "_")
            with open(filename, "w") as f:
                f.writelines("\n".join(lines))

if __name__ == "__main__":
    # Get arguments for score
    parser = argparse.ArgumentParser(description="Run the script with nutrient density score of interest and threshold.")
    parser.add_argument('nds', type=str, help="Nutrition desnsity score. ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA")
    # All (>0; restaurants), 30% (>3; restaurants), 50% (>15; restaurants), and 70% (>80; restaurants)
    parser.add_argument('threshold', type=int, help="All, 30%, 50%, and 70% threshold for sensitivity analysis. ex) 0, 30, 50, 70")
    args = parser.parse_args()
    # Read FEND datasets
    county_df = pd.read.csv(f"../data/files/FEND_{args.nds}.csv", dtype={'CountyFIPS':str})
    ct_df = pd.read.csv(f"../data/files/FEND_{args.nds}_ct.csv", dtype={'TractFIPS':str})
    # 50% thresholding (>15): main result 
    county_df_main = thresholding(args.threshold, county_df)
    # Get results # Plot figures
    results = GetResults(df=county_df_main)
    results.correlation_heatmap()
    results.summary_stats(df_all=county_df)
    results.metro_rural_test()
    results.decile_analysis()
    results.ee_barplot(df=pd.read.csv("main_linear_model_table.csv")) # Import linear model results as table
    results.extra_lm(df_ct=ct_df, ruca=pd.read_excel('ruca2010revised.xlsx').iloc[1:,0:5], th=args.threshold) # census-tracts
