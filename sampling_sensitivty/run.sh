source ../venv/bin/activate

source ../venv/bin/activate

cat << 'EOF' | xargs -P 6 -I {} bash -c "python generate_data.py {}"
-s cma_single -f ela -n 100 -r 30
-s uniform -f ela -n 10 -r 30
-s uniform -f ela -n 25 -r 30
-s uniform -f ela -n 50 -r 30
-s uniform -f ela -n 75 -r 30
-s uniform -f ela -n 100 -r 30
-s ilhs -f ela -n 10 -r 30
-s ilhs -f ela -n 25 -r 30
-s ilhs -f ela -n 50 -r 30
-s ilhs -f ela -n 75 -r 30
-s ilhs -f ela -n 100 -r 30
-s lhs -f ela -n 10 -r 30
-s lhs -f ela -n 25 -r 30
-s lhs -f ela -n 50 -r 30
-s lhs -f ela -n 75 -r 30
-s lhs -f ela -n 100 -r 30
-s sobol -f ela -n 10 -r 30
-s sobol -f ela -n 25 -r 30
-s sobol -f ela -n 50 -r 30
-s sobol -f ela -n 75 -r 30
-s sobol -f ela -n 100 -r 30
EOF


#py generate_data.py -s cma_single -f ela -n 100 -r 30
#
#py generate_data.py -s uniform -f ela -n 10 -r 30
#py generate_data.py -s uniform -f ela -n 25 -r 30
#py generate_data.py -s uniform -f ela -n 50 -r 30
#py generate_data.py -s uniform -f ela -n 75 -r 30
#py generate_data.py -s uniform -f ela -n 100 -r 30
#
#py generate_data.py -s ilhs -f ela -n 10 -r 30
#py generate_data.py -s ilhs -f ela -n 25 -r 30
#py generate_data.py -s ilhs -f ela -n 50 -r 30
#py generate_data.py -s ilhs -f ela -n 75 -r 30
#py generate_data.py -s ilhs -f ela -n 100 -r 30
#
#py generate_data.py -s lhs -f ela -n 10 -r 30
#py generate_data.py -s lhs -f ela -n 25 -r 30
#py generate_data.py -s lhs -f ela -n 50 -r 30
#py generate_data.py -s lhs -f ela -n 75 -r 30
#py generate_data.py -s lhs -f ela -n 100 -r 30
#
#py generate_data.py -s sobol -f ela -n 10 -r 30
#py generate_data.py -s sobol -f ela -n 25 -r 30
#py generate_data.py -s sobol -f ela -n 50 -r 30
#py generate_data.py -s sobol -f ela -n 75 -r 30
#py generate_data.py -s sobol -f ela -n 100 -r 30
