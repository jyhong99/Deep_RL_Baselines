import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_train_result(project_name, epoch_logger, window=50, show_graphs=True):
    logger_df = pd.DataFrame(epoch_logger)
    if logger_df.empty:
        print("Logger data is empty. No data to plot.")
        return

    clean_df = logger_df.dropna(subset=['timesteps', 'max_ep_ret', 'mean_ep_ret', 'max_ep_len', 'mean_ep_len'])
    fig, axs = plt.subplots(3, 2, figsize=(16, 15))

    if 'timesteps' in clean_df.columns and 'max_ep_ret' in clean_df.columns:
        axs[0, 0].plot(clean_df['timesteps'], clean_df['max_ep_ret'], 'b-')
        axs[0, 0].set_title("Max Episode Return over Timesteps")
        axs[0, 0].set_xlabel("Timesteps")
        axs[0, 0].set_ylabel("Max Return")
        axs[0, 0].grid(axis='y')

    if 'timesteps' in clean_df.columns and 'mean_ep_ret' in clean_df.columns:
        axs[0, 1].plot(clean_df['timesteps'], clean_df['mean_ep_ret'], 'r-')
        axs[0, 1].set_title("Mean Episode Return over Timesteps")
        axs[0, 1].set_xlabel("Timesteps")
        axs[0, 1].set_ylabel("Return")
        axs[0, 1].grid(axis='y')

    if 'timesteps' in clean_df.columns and 'mean_ep_ret' in clean_df.columns:
        cumulative_mean = clean_df['mean_ep_ret'].expanding().mean()
        axs[1, 0].plot(clean_df['timesteps'], cumulative_mean, 'g-')
        axs[1, 0].set_title("Cumulative Mean Episode Return")
        axs[1, 0].set_xlabel("Timesteps")
        axs[1, 0].set_ylabel("Cumulative Mean Return")
        axs[1, 0].grid(axis='y')

    if 'timesteps' in clean_df.columns and 'mean_ep_ret' in clean_df.columns and len(clean_df['mean_ep_ret']) >= window:
        steps = clean_df['timesteps']
        ep_ret_values = clean_df['mean_ep_ret']
        mean_ep_ret = ep_ret_values.rolling(window=window).mean()
        std_ep_ret = ep_ret_values.rolling(window=window).std()

        axs[1, 1].plot(steps, mean_ep_ret, 'y-')
        axs[1, 1].fill_between(steps, mean_ep_ret - std_ep_ret, mean_ep_ret + std_ep_ret, color='y', alpha=0.1)
        axs[1, 1].set_title(f"Rolling Mean and Std of Return (Window={window})")
        axs[1, 1].set_xlabel("Timesteps")
        axs[1, 1].set_ylabel("Average Return")
        axs[1, 1].grid(axis='y')

    if 'timesteps' in clean_df.columns and 'max_ep_len' in clean_df.columns:
        axs[2, 0].plot(clean_df['timesteps'], clean_df['max_ep_len'], 'b-')
        axs[2, 0].set_title("Max Episode Length over Timesteps")
        axs[2, 0].set_xlabel("Timesteps")
        axs[2, 0].set_ylabel("Max Length")
        axs[2, 0].grid(axis='y')

    if 'timesteps' in clean_df.columns and 'mean_ep_len' in clean_df.columns:
        axs[2, 1].plot(clean_df['timesteps'], clean_df['mean_ep_len'], 'r-')
        axs[2, 1].set_title("Mean Episode Length over Timesteps")
        axs[2, 1].set_xlabel("Timesteps")
        axs[2, 1].set_ylabel("Length")
        axs[2, 1].grid(axis='y')

    fig.tight_layout()

    if project_name:
        save_path = f'./log/{project_name}'
        os.makedirs(save_path, exist_ok=True)
        plot_save_path = os.path.join(save_path, f'{project_name}_plot.png')
        plt.savefig(plot_save_path)
        print(f'Plot has been saved at: {plot_save_path}')

    if show_graphs:
        plt.show()


def plot_epoch_result(project_name, epoch_logger, window=50, show_graphs=True):
    expanded_logger = []
    for entry in epoch_logger:
        if 'result' in entry:
            result_data = entry.pop('result')       
            entry.update(result_data)
        expanded_logger.append(entry)

    logger_df = pd.DataFrame(expanded_logger)
    logger_df.interpolate(method='linear', inplace=True)
    
    excluded_keys = [
        'timesteps', 
        'number_of_eps', 
        'max_ep_ret', 
        'max_ep_len', 
        'mean_ep_ret', 
        'mean_ep_len', 
        'agent_timesteps', 
        'td_error'
    ]
    
    result_keys = [key for key in logger_df.columns if key not in excluded_keys]
    num_metrics = len(result_keys)
    num_rows = num_metrics
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 5 * num_rows))

    if num_metrics == 1:
        axs = [axs]

    for idx, key in enumerate(result_keys):
        steps = logger_df['timesteps']
        values = logger_df[key]

        axs[idx][0].plot(steps, values, label=f"{key} (raw)", alpha=0.5)
        axs[idx][0].set_title(f"{key}")
        axs[idx][0].set_xlabel("Timesteps")
        axs[idx][0].set_ylabel(key)
        axs[idx][0].grid(axis='y')

        if len(values) >= window:
            rolling_mean = values.rolling(window=window, min_periods=1).mean()
            rolling_std = values.rolling(window=window, min_periods=1).std()

            axs[idx][1].plot(steps, rolling_mean, color='y')
            axs[idx][1].fill_between(steps, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                                    color='y', alpha=0.1)
        axs[idx][1].set_title(f"{key} (Mean & Std of {window} data)")
        axs[idx][1].set_xlabel("Timesteps")
        axs[idx][1].set_ylabel(key)
        axs[idx][1].grid(axis='y')

        axs[idx][0].legend(loc="best")
        axs[idx][1].legend(loc="best")

    fig.tight_layout()

    if project_name:
        save_path = f'./log/{project_name}'
        os.makedirs(save_path, exist_ok=True)
        plot_save_path = os.path.join(save_path, f'{project_name}_epoch_plot.png')
        plt.savefig(plot_save_path)
        print(f'Plot has been saved at: {plot_save_path}')

    if show_graphs:
        plt.show()