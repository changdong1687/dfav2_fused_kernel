python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1024 -seqt 0 
echo "======== no-full seqv 1024 seqt 0 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1088 -seqt 0 
echo "======== no-full seqv 1088 seqt 0 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 0 -seqt 1024
echo "======== no-full seqv 0 seqt 1024 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 0 -seqt 1560
echo "======== no-full seqv 0 seqt 1560 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1024 -seqt 256 
echo "======== no-full seqv 1024 seqt 333 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1024 -seqt 333
echo "======== no-full seqv 1024 seqt 333 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1847 -seqt 256
echo "======== no-full seqv 1847 seqt 256 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 0 -seqv 1847 -seqt 333
echo "======== no-full seqv 1847 seqt 333 ========\n"

echo "\n========================= some full =======================\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1024 -seqt 0 
echo "======== no-full seqv 1024 seqt 0 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1088 -seqt 0 
echo "======== no-full seqv 1088 seqt 0 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 0 -seqt 1024
echo "======== no-full seqv 0 seqt 1024 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 0 -seqt 1560
echo "======== no-full seqv 0 seqt 1560 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1024 -seqt 256 
echo "======== no-full seqv 1024 seqt 333 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1024 -seqt 333
echo "======== no-full seqv 1024 seqt 333 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1847 -seqt 256
echo "======== no-full seqv 1847 seqt 256 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 2 -seqv 1847 -seqt 333
echo "======== no-full seqv 1847 seqt 333 ========\n"

echo "\n========================= all full =======================\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1024 -seqt 0 
echo "======== no-full seqv 1024 seqt 0 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1088 -seqt 0 
echo "======== no-full seqv 1088 seqt 0 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 0 -seqt 1024
echo "======== no-full seqv 0 seqt 1024 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 0 -seqt 1560
echo "======== no-full seqv 0 seqt 1560 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1024 -seqt 256 
echo "======== no-full seqv 1024 seqt 333 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1024 -seqt 333
echo "======== no-full seqv 1024 seqt 333 ========\n"

python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1847 -seqt 256
echo "======== no-full seqv 1847 seqt 256 ========\n"
python /root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/test_headwise_arrow.py --headwin -nh 8 -fn 8 -seqv 1847 -seqt 333
echo "======== no-full seqv 1847 seqt 333 ========\n"