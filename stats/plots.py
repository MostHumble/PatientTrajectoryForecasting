import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
import numpy as np


def get_admission_stats(subject_id_adm_map, num_visits=15):
   # Count the number of admissions per subject
    num_admissions_less = [len(admissions) for admissions in subject_id_adm_map.values() if len(admissions) < num_visits]
    num_admissions_more = [len(admissions) for admissions in subject_id_adm_map.values() if len(admissions) > num_visits]
    
    counts_less = dict(Counter(num_admissions_less))
    counts_more = dict(Counter(num_admissions_more))
    counts_more_sorted = sorted(counts_more.items())

    x, y = zip(*counts_more_sorted)
    cumulative = [sum(y[:i+1]) for i in range(len(y))]
    
    return counts_less, (x, cumulative)


def plot_all_distributions(steps_data, num_visits=15, save=False, dpi=600, prefix=''):
    fig, axs = plt.subplots(2, 1, figsize=(20, 16))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot the first distribution
    for i, (step, (counts_less, _)) in enumerate(steps_data.items()):
        color = colors[i % len(colors)]
        axs[0].bar(counts_less.keys(), counts_less.values(), color=color, alpha=0.5, label=step)
    
    axs[0].set_xlabel('Number of visits')
    axs[0].set_ylabel('Number of Subjects')
    axs[0].set_title(f'Distribution of Number of Subjects per Number of Visits (< {num_visits})')
    axs[0].set_xlim(0, num_visits)
    axs[0].set_xticks(range(1, num_visits))
    axs[0].legend()
    
    for step, (counts_less, _) in steps_data.items():
        for x, y in counts_less.items():
            axs[0].text(x, y, str(y), ha='center', va='bottom')

    # Plot the second distribution
    for i, (step, (_, (x, cumulative))) in enumerate(steps_data.items()):
        color = colors[i % len(colors)]
        axs[1].plot(x, cumulative, marker='o', linestyle='-', color=color, alpha=0.5, label=step)
    
    axs[1].set_xlabel('Number of visits')
    axs[1].set_ylabel('Number of Subjects')
    axs[1].set_title(f'Cumulative Distribution of Number of Subjects per Number of Visits (> {num_visits})')
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].legend()

    plt.tight_layout()
          
    if save:
        plt.savefig(f'./stats/figures/distribution_subject_num_visits_combined_{prefix}.png', dpi=dpi)
    plt.show()


def plot_admission_distribution(subject_id_adm_map, num_visits = 15, save = False, dpi = 600, prefix :str = '', title :str = ''):
    # Count the number of admissions per subject
    counts_less, (x, cumulative) = get_admission_stats(subject_id_adm_map, num_visits)

    fig, axs = plt.subplots(1, 2, figsize = (20, 8))

    # Plot the first distribution
    axs[0].bar(counts_less.keys(), counts_less.values(), color='green')  # Change color to blue for all bars
    axs[0].bar(1, counts_less.get(1, 0), color='red')  # Change color to red for the first bar
    axs[0].set_xlabel('Number of visits')
    axs[0].set_ylabel('Number of Subjects')
    axs[0].set_title(f'Distribution of Number of Subject per Number of Visits ({num_visits} <)')
    axs[0].set_xlim(0, num_visits)
    axs[0].set_xticks(range(1, num_visits))
    axs[0].set_yticks([])

    for i, j in counts_less.items():
        axs[0].text(i, j, str(j), ha = 'center', va = 'bottom')

    # Plot the second distribution    
    axs[1].set_xlabel('Number of visits')
    axs[1].set_ylabel('Number of Subjects')
    axs[1].set_title(f'Cumulative distribution of Number of Subject per Number of Visits (> {num_visits} )')
    axs[1].plot(x, cumulative, marker='o', linestyle='-', color='red')
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True)) # Set the y-axis to integer values
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
          
    if save:
        plt.savefig(f'./stats/figures/distribution_subject_num_visits_{prefix}_{num_visits}.png', dpi=dpi)
    plt.show()


def plot_sequences_lenghts(lengths_source_sequences, lengths_target_sequences):
    # Plot histograms
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(lengths_source_sequences, bins=10, alpha=0.7, color='blue')
    plt.title('Source Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(lengths_target_sequences, bins=10, alpha=0.7, color='green')
    plt.title('Target Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_percentiles(lengths, title, color):
    plt.figure(figsize=(10, 6))
    percentiles = np.percentile(lengths, np.arange(0, 101, 1))
    plt.plot(percentiles, np.arange(0, 101, 1), label=title, color=color)
    plt.xlabel('Sequence Length')
    plt.ylabel('Percentile')
    plt.title('Percentile Plot of Sequence Lengths')
    plt.legend()
    plt.grid(True)
    plt.show()

