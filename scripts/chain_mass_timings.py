import pandas as pd
import matplotlib.pyplot as plt


def main():
    fig_path = "scripts/figures"
    # Load the timings
    timings_df = pd.read_csv(f"{fig_path}/chain_mass_timings.csv", index_col=0)

    # Plot the timings
    plt.figure()
    for key in timings_df.keys():
        plt.plot(timings_df.index, timings_df[key], label=key)
    plt.ylabel("Time [s]")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{fig_path}/chain_mass_timings")
    plt.show()


if __name__ == "__main__":
    main()
