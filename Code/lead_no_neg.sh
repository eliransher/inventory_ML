#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 15000
source /home/eliransc/projects/def-dkrass/eliransc/mom_match/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/mom_analysis/code/fit_moms_main.py