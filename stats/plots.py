import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator

def plot_admission_distribution(subject_id_adm_map, num_visits = 15, save = False, dpi = 600, prefix :str = '', title :str = ''):
    # Count the number of admissions per subject
    num_admissions_less = [len(admissions) for admissions in subject_id_adm_map.values() if len(admissions) < num_visits]
    num_admissions_more = [len(admissions) for admissions in subject_id_adm_map.values() if len(admissions) > num_visits]
    
    counts_less = dict(Counter(num_admissions_less))
    counts_more = dict(Counter(num_admissions_more))
    counts_more_sorted = sorted(counts_more.items())

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

    for x, y in counts_less.items():
        axs[0].text(x, y, str(y), ha = 'center', va = 'bottom')

    # Plot the second distribution
    x, y = zip(*counts_more_sorted)
    cumulative = [sum(y[:i+1]) for i in range(len(y))]
    
    axs[1].set_xlabel('Number of visits')
    axs[1].set_ylabel('Number of Subjects')
    axs[1].set_title(f'Cumulative distribution of Number of Subject per Number of Visits (> {num_visits} )')
    axs[1].plot(x, cumulative, marker='o', linestyle='-', color='red')
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True)) # Set the y-axis to integer values
    if title:
        plt.suptitle(title)
    plt.tight_layout()
          
    if save:
        plt.savefig(f'./stats/figures/distribution_subject_num_visits_{prefix}_{num_visits}.png', dpi=dpi)
    plt.show()
