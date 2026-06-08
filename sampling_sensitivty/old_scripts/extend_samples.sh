source ../venv/bin/activate

cat << 'EOF' | xargs -P 10 -I {} bash -c "python extend_samples.py {}"
-s cma_random -n 100 -r 30 --input-dir ../../data/samples
-s lhs -n 10 -r 30 --input-dir ../../data/samples
-s lhs -n 25 -r 30 --input-dir ../../data/samples
-s lhs -n 50 -r 30 --input-dir ../../data/samples
-s lhs -n 75 -r 30 --input-dir ../../data/samples
-s lhs -n 100 -r 30 --input-dir ../../data/samples
-s sobol -n 10 -r 30 --input-dir ../../data/samples
-s sobol -n 25 -r 30 --input-dir ../../data/samples
-s sobol -n 50 -r 30 --input-dir ../../data/samples
-s sobol -n 75 -r 30 --input-dir ../../data/samples
-s sobol -n 100 -r 30 --input-dir ../../data/samples
-s uniform -n 10 -r 30 --input-dir ../../data/samples
-s uniform -n 25 -r 30 --input-dir ../../data/samples
-s uniform -n 50 -r 30 --input-dir ../../data/samples
-s uniform -n 75 -r 30 --input-dir ../../data/samples
-s uniform -n 100 -r 30 --input-dir ../../data/samples
-s ilhs -n 10 -r 30 --input-dir ../../data/samples
-s ilhs -n 25 -r 30 --input-dir ../../data/samples
-s ilhs -n 50 -r 30 --input-dir ../../data/samples
-s ilhs -n 75 -r 30 --input-dir ../../data/samples
-s ilhs -n 100 -r 30 --input-dir ../../data/samples
EOF