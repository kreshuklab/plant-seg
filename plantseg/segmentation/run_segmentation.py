

if __name__ == "__main__":
    import glob
    import os

    gasp_enviroment = 'gasp'  # full env name /home/lcerrone_local/miniconda3/envs/gasp/bin/python
    multicut_enviroment = 'multi_cut'  # full env name  /home/lcerrone_local/miniconda3/envs/multi_cut/bin/python

    # grab env
    full_env = os.environ['_']
    env_name = full_env[full_env.find("envs/") + 5:full_env.find("/bin/python")]
    bypass_flag = None  # If this is "gasp" or "multi_cut" bypass the automatic env matching

    all_paths = sorted(glob.glob("/mnt/localdata0/lcerrone/FOR_paper/COS_ch_meristem/ds1_predicitons_eqods3/unetds1x/*tions.h5"))

    # Select the right set of baselines to run
    if env_name == gasp_enviroment or bypass_flag == gasp_enviroment:
        from gasp import GaspFromPmaps
        from watershed import DtWatershedFromPmaps


        baselines_env = {"gasp_average": GaspFromPmaps(save_directory="GASP",
                                                       gasp_linkage_criteria='average',
                                                       gasp_beta_bias=0.6,
                                                       run_ws=True,
                                                       ws_threshold=0.6,
                                                       ws_minsize=50,
                                                       ws_sigma=1.0,
                                                       post_minsize=50,  # 50 best ds 3
                                                       n_threads=6),
                         "gasp_mutexwatershed": GaspFromPmaps(save_directory="MutexWatershed",
                                                              gasp_linkage_criteria='mutex_watershed',
                                                              gasp_beta_bias=0.6,
                                                              run_ws=True,
                                                              ws_threshold=0.5,
                                                              ws_minsize=30,
                                                              ws_sigma=1.0,
                                                              post_minsize=50,  # 50 best ds 3
                                                              n_threads=6),
        }

    elif env_name == multicut_enviroment or bypass_flag == multicut_enviroment:
        from multicut import MulticutFromPmaps
        from randomwalker import DtRandomWalkerFromPmaps

        baselines_env = {"multicut": MulticutFromPmaps(save_directory="MultiCut 3Dws",
                                                       multicut_beta=0.6,
                                                       run_ws=True,
                                                       ws_2D=True,
                                                       ws_threshold=0.5,
                                                       ws_minsize=100,
                                                       ws_sigma=2.0,
                                                       post_minsize=100,
                                                       n_threads=6)}

    else:
        raise NotImplementedError

    for path in all_paths:
        for algorithm, func in baselines_env.items():
            print("Run %s on: %s" % (algorithm, path))
            func(path)
