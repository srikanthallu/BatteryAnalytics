#!/bin/bash

function create_input {
	cat << _EOF_ > ${SCRIPT_FILENAME}
#!/bin/bash

source ../../env/bin/activate

python lstm_battery_analytics.py --window $WINDOW --horizon $HORIZON --hidden-dim $HIDDEN_DIM --n-epochs $N_EPOCHS --n-split $N_SPLIT --training-filter $TRAINING_FILTER --model-id $MODEL_ID --features $FEATURES --infile $FILENAME

_EOF_
}


WINDOW_LIST=(150)
HORIZON_LIST=(300)
HIDDEN_DIM_LIST=(2 4 8 16)
N_EPOCHS=10
TRAINING_FILTER_LIST=(none savgol isotonic_regression)
MODEL_ID_LIST=( $(seq 1 5) )
FEATURES_LIST=(
	"'2000 Pounds [Pounds]' 'Voltage [V]'"
	"'2000 Pounds [Pounds]' 'Voltage [V]' '2000 Pounds [Pounds] (gradient 1)' 'Voltage [V] (gradient 1)'"
	"'2000 Pounds [Pounds]' 'Voltage [V]' '2000 Pounds [Pounds] (savgol 1)' 'Voltage [V] (savgol 1)'"
)
FILENAME="../data/mechanical_loading_data.csv.gz"

II=0
COUNT=0
for N_SPLIT in {0..34}; do
	II=0
	DIRNAME="./inp_$(printf "%02d" ${N_SPLIT})"
	mkdir -p "${DIRNAME}"
	for TRAINING_FILTER in "${TRAINING_FILTER_LIST[@]}"; do
		for MODEL_ID in "${MODEL_ID_LIST[@]}"; do
			for FEATURES in "${FEATURES_LIST[@]}"; do
				for HIDDEN_DIM in "${HIDDEN_DIM_LIST[@]}"; do
					for WINDOW in "${WINDOW_LIST[@]}"; do
						for HORIZON in "${HORIZON_LIST[@]}"; do
							BASENAME="work_$(printf "%03d" $II).inp"
							SCRIPT_FILENAME="${DIRNAME}/${BASENAME}"
							create_input
							II=$((II+1))
						done
					done
				done
			done
		done
	done
	sleep 5

	COUNT=$((COUNT+II))
	NUMFILES=$(($II - 1))
	echo -n "work ${II} : "
	if [ $NUMFILES -ge 0 ]; then
		res=$(sbatch --array=0-$NUMFILES $DEPENDENCY --export=DIRNAME="${DIRNAME}" array_job.sh)
		echo $res
		JOBID=$(echo $res | awk -F" " '{print $NF}')
		#DEPENDENCY="--dependency=afterany:$JOBID"
	else
		echo "No files found."
	fi
done
echo "total ${COUNT}"
