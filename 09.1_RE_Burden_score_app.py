import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

st.set_page_config(layout="centered")

# Function to calculate top percentile based on input parameters
def top_percentile(df, columns_to_check, covariates, id_column, percentile=0.8):
    percentile_df = df[[id_column]]
    covariates_df = df[covariates]
    
    result_cols = []
    for col in columns_to_check:
        median_value = df[col].median()
        if median_value > 0:
            threshold = df[col].quantile(percentile)
            result_col = (df[col] >= threshold).astype(int)
        else:
            result_col = pd.Series([0] * len(df), index=df.index)  # Set to 0 if median <= 0
        result_cols.append(result_col)
    
    percentile_df = pd.concat([percentile_df] + result_cols, axis=1)
    percentile_df['score'] = percentile_df.drop(id_column, axis=1).sum(axis=1)
    percentile_df = pd.merge(percentile_df, covariates_df, how='inner', on=id_column)
    
    return percentile_df

# Function to standardize the score
def rescale_to_one_sd(df, column_name):
    std_dev = df[column_name].std()
    df['scaled_score'] = (df[column_name]) / std_dev
    return df['scaled_score']

def plot_partial_effects(cph, df_score):#, p_value_score):
    fig, ax = plt.subplots(figsize=(12, 8))
    cph.plot_partial_effects_on_outcome(covariates=['scaled_score'],
                                        values=[df_score['scaled_score'].mean()],
                                        ax=ax,
                                        plot_baseline=False)
    plt.title(f'Partial Effects of TE Score on Survival \n(p-value for score') # = {p_value_score:.2e})')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.xlim(0, 40)
    plt.legend().remove()
    return fig

def plot_model(cph):
    fig, ax = plt.subplots(figsize=(8, 6))
    cph.plot(ax=ax)
    return fig

def quantile_plots(df, score_col, survival_time_col, vital_status_col, num_quantiles):
    quantile_values, bins = pd.qcut(df[score_col], q=num_quantiles, labels=False, retbins=True, duplicates='drop')
    bins = np.unique(bins)  # Ensure unique bin edges
    
    # Assign quantile groups
    df['quantile_group'] = pd.cut(df[score_col], bins=bins, labels=[f'Q{i+1}' for i in range(len(bins)-1)])
    
    # Kaplan-Meier Fitter
    kmf = KaplanMeierFitter()
    
    # Plot the Kaplan-Meier curves
    plt.figure(figsize=(10, 6))
    
    for name, grouped_df in df.groupby('quantile_group'):
        kmf.fit(durations=grouped_df[survival_time_col], event_observed=grouped_df[vital_status_col], label=name)
        kmf.plot_survival_function(show_censors=True, ci_show=False)  # ci_show=False to hide confidence intervals
    
    plt.title(f'Kaplan-Meier Curve by Score Quantiles')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend(title='Quantiles')
    st.pyplot(plt)

def subset_by_family(df, gene_info, selected_family, id_column):
    list1 = list(gene_info[gene_info['re_family'] == selected_family]['re'])
    list2 = list(df.columns)
    overlap = list(set(list1) & set(list2))
    family_df = df[overlap]
    family_df["IID"] = df[id_column]
    family_df = pd.merge(family_df, df[['IID','survival_months','deceased','age','sex']], how='inner', on=id_column)
    return family_df

# Function to display the Streamlit app
def main():
    st.title('TE Burden score Analysis')
    
    # Add instructions
    st.markdown("""
    ## Instructions
    
    ### File formatting:
    * The RE count file and the RE info file should be in .csv format
    * At the **Non-RE columns** anything that is not an RE (including, ID, age, sex, etc)
    * The file should have a column named `survival_months` and `deceased`. 
        * The column `deceased` should be a binary column `[0,1]` to denote subjects who are deceased 
    
    ### Parameters
    * Under the *Select RE family* slect one family or select all to create a model and plots without subsetting by family
    * The *Select percentile* is the threshold at which the program will classify subjects as having `0s` or `1s` and that will be used for the score.
    * 'Select Number for Score Quantiles is the number of quantiles that the score will be sectioned into and the Quantiles plot will be created based on this
    
    **Notes:**  \n
    * *The model will adjust for age and sex and the coefficients will be displayed under the plot*
    * *The program will display a preview of the data that is being used for the quantile plots*
    
    """)

    # Upload file 
    
    st.sidebar.subheader("Upload files")
    uploaded_file = st.sidebar.file_uploader("Upload your Count file", type=['csv'])
    rep_info_file = st.sidebar.file_uploader("Upload your RE info file", type=['csv'])
    
    if uploaded_file is not None and rep_info_file is not None:
        re_df = pd.read_csv(uploaded_file)
        rep_info = pd.read_csv(rep_info_file)
        
        
        
        id_column = st.sidebar.selectbox('Select ID Column', re_df.columns)
        
        st.sidebar.subheader('Select all the Non-RE Columns')
        
        columns_to_exclude = st.sidebar.multiselect('Non-RE variables', re_df.columns)
        
        st.sidebar.subheader('Analysis Parameters')
        
        re_family = st.sidebar.selectbox('Select an RE family', ['All'] + list(rep_info['re_family'].unique()))
        
        percentile = st.sidebar.slider('Select Percentile Threshold', min_value=0.1, max_value=0.9, value=0.8, step=0.01)
        
        num_quantiles = st.sidebar.number_input('Select Number for Score Quantiles', min_value=2, max_value=10, value=5, step=1)
        
        if st.sidebar.button('Run Analysis'):
            if re_family != 'All':
                re_df = subset_by_family(re_df, rep_info, re_family, id_column)
            
            columns_to_check = list(set(re_df.columns) - set(columns_to_exclude + [id_column]))
            
            df_score = top_percentile(df=re_df, columns_to_check=columns_to_check, covariates=columns_to_exclude, id_column=id_column, percentile=percentile)
            df_score = pd.get_dummies(df_score, columns=['sex'], drop_first=True)
            df_score['scaled_score'] = rescale_to_one_sd(df_score, 'score')
            
            cph = CoxPHFitter()
            cph.fit(df_score, duration_col='survival_months', event_col='deceased', formula='scaled_score + sex_male + age')
                        
            #p_value_score = cph.summary.loc['scaled_score', 'p']
            
            # DISPLAY OUTPUTS
            st.subheader(f'Partial Effects of TE Score on Survival for family {re_family}')
           #st.write(f'(p-value for score'))# = {p_value_score:.2e})')
            st.pyplot(plot_partial_effects(cph, df_score)) #, p_value_score))  
            
            st.subheader('Cox Proportional Hazards Model')
            st.write(cph.summary) 
            st.pyplot(plot_model(cph))
            
            if num_quantiles != 0: 
                st.subheader('Kaplan-Meier Curve by Score Quantiles')
                quantile_plots(df_score, 'scaled_score', 'survival_months', 'deceased', num_quantiles)
            
            # Data Preview
            st.subheader(f"Data Preview for {re_family} family")
            st.write(df_score.head(20))

if __name__ == '__main__':
    main()
