from rlmpc.examples.chain_mass import main


def test_main():
    main(np_test=10, plot=False, save_timings=False)
    assert True


if __name__ == "__main__":
    test_main()
