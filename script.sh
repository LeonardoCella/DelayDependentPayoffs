for fra_top in 0.1 0.2
    do
    for gamma in 0.7 0.999 
        do
        for delta in 0.1
            do
            for d in 5
                do
                for bin in 1
                    do
                    for k in 4 6 8
                        do
                        declare -i tau
                        tau=$((${d}+1))
                        printf "===============\n===============\npython3.6 -u run.py --gamma $gamma -T 1000000 --max_delay $d --tau $tau -k $k --fra_top $fra_top --delta $delta --n_rep 5 -v 1 --bin $bin -s 1\n===============\n==============="  
                        if [ $bin == 1 ]
                            then
                            python3.6 run.py --gamma $gamma -T 1000000 --max_delay $d --tau $tau -k $k --fra_top $fra_top --delta $delta --n_rep 5 -v 1 --bin $bin -s 1 --stage 2
                        else
                            python3.6 run.py --gamma $gamma -T 1000000 --max_delay $d --tau $tau -k $k --fra_top $fra_top --delta $delta --n_rep 1 -v 1 --bin $bin -s 1
                        fi
                    done
                done
            done
        done
    done
done
