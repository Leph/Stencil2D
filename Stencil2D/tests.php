<?php

$MATRIX_SIZE = 8192;
$MAX_ITERATION = 50;
$STEP_ITERATION = 5;
$STEP_YDIM_GPU = 256;
$LOOP_AVG = 5;
$OUTPUT_FILE = 'graph3.dat';

system("echo \"#MATRIX_SIZE : $MATRIX_SIZE\" > $OUTPUT_FILE");
$date = system("date");
system("echo \"#START : $date\" >> $OUTPUT_FILE");
system("echo \"#num_iteration\tydim_gpu\tspeedup\" >> $OUTPUT_FILE");

for ($num_iteration=1 ; $num_iteration<=$MAX_ITERATION ; $num_iteration+=$STEP_ITERATION) {
	for ($ydim_gpu=0 ; $ydim_gpu<=$MATRIX_SIZE ; $ydim_gpu+=$STEP_YDIM_GPU) {
		$defines = "\"-DQUIET=1 -DYDIM=$MATRIX_SIZE -DXDIM=$MATRIX_SIZE -DNUM_ITERATION=$num_iteration -DYDIM_GPU=$ydim_gpu\"";
		system("make DEFINES=$defines");
		$result = 0.0;
		for ($i=1;$i<=$LOOP_AVG;$i++) {
			$result += system("./stencil");
		}
		$result = $result/$LOOP_AVG;
		system("echo \"$num_iteration\t$ydim_gpu\t$result\" >> $OUTPUT_FILE");
	}
}

$date = system("date");
system("echo \"#END : $date\" >> $OUTPUT_FILE");

