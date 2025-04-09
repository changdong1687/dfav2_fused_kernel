echo "=================4096 + 512=================="
python mytest/measure_time_trans.py -m 2  -k 1
echo ""
echo "=================4096 + 512=================="

python mytest/measure_time_trans.py -m 2  -k 2
echo ""
echo "=================4096 + 512=================="
python mytest/measure_time_trans.py -m 2  -k 3
echo ""

echo "=================16384 + 512=================="
python mytest/measure_time_trans.py -m 3  -k 1
echo "=================16384 + 512=================="
python mytest/measure_time_trans.py -m 3  -k 2
echo "=================16384 + 512=================="
python mytest/measure_time_trans.py -m 3  -k 3
