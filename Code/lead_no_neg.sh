#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 15000
source /home/eliransc/projects/def-dkrass/eliransc/mom_match/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/inventory_ML/Code/lead_time_no_negative_object.py