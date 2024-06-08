#!/bin/sh

# python run.py --model mlp --nQuery 1000 --data FILE --alg vessal --nQuery 18

# python run.py --model mlp --nQuery 1000 --data FILE --alg stream_rand --nQuery 18

DEBUG_ENABLED=1 python run.py --model mlp --nQuery 2000 --data FILE --alg stream_rand_torch 
#--nQuery 18