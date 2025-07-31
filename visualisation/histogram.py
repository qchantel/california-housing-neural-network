import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data.data_loader import drop_outliers, load_data, preprocess_data

def plot_house_values_histogram(df=None, bins=50, figsize=(12, 8), save_path=None):
    """
    Create a histogram of house values from the California housing dataset.
    
    Args:
        df (pd.DataFrame, optional): Pre-loaded dataframe. If None, loads data from file.
        bins (int): Number of bins for the histogram
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Load data if not provided
    if df is None:
        return None
    
    # Extract house values
    house_values = df['median_house_value']
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create histogram
    plt.hist(house_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Median House Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of House Values in California Housing Dataset', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = house_values.mean()
    median_val = house_values.median()
    std_val = house_values.std()
    
    stats_text = f'Mean: ${mean_val:,.0f}\nMedian: ${median_val:,.0f}\nStd Dev: ${std_val:,.0f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format x-axis to show dollar amounts
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ Histogram saved to {save_path}')
    else:
        plt.show()
    
    return plt.gca()

def correlation_matrix(df=None, figsize=(12, 10), save_path=None):
    """
    Create and display a correlation matrix heatmap for the California housing dataset.
    
    Args:
        df (pd.DataFrame, optional): Pre-loaded dataframe. If None, loads data from file.
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Load data if not provided
    if df is None:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Set ticks and labels
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Find the index of median_house_value column
    target_col = 'median_house_value'
    if target_col in corr_matrix.columns:
        target_idx = corr_matrix.columns.get_loc(target_col)
    else:
        target_idx = None
    
    # Add correlation values as text and highlight median_house_value
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            
            # Highlight median_house_value column and row
            if target_idx is not None and (i == target_idx or j == target_idx):
                # Add a border around the cell
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                   fill=False, edgecolor='yellow', linewidth=3, alpha=0.8)
                plt.gca().add_patch(rect)
                # Make text bold for highlighted cells
                fontweight = 'bold'
                fontsize = 12
            else:
                fontweight = 'normal'
                fontsize = 10
            
            plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontsize=fontsize, fontweight=fontweight)
    
    # Add title
    plt.title('Correlation Matrix - California Housing Dataset', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ Correlation matrix saved to {save_path}')
    else:
        plt.show()
    
    return plt.gca()

if __name__ == "__main__":
    # Load data
    df = load_data()
    df_encoded = preprocess_data(df, one_hot_encode=False)
    # df_encoded = drop_outliers(df_encoded)
    
    # Create basic histogram
    print("Creating histogram of house values...")
    plot_house_values_histogram(df_encoded)

    print("Creating correlation matrix...")
    correlation_matrix(df_encoded)
    