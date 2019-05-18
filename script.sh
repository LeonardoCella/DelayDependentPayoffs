for gamma in 0.2 0.5 0.7 0.999
    do
    for fra_top in 0.1 0.3 0.5
        do
        for delta in 0.1
            do
            for ub in 1 3
                do
                for d in 5
                    do
                    for bin in 0 1
                        do
                        for k in 4 6
                            do
                            declare -i tau
                            tau=$((${d}+${ub}+1))
                            printf "===============\n===============\npython3.6 -u run.py --gamma $gamma -T 200000 --max_delay $d -d $tau -k $k --fra_top $fra_top --delta $delta --delay_ub $ub --n_rep 5 -v 1 --bin $bin -s 1\n===============\n==============="  
                            if [ $bin == 1 ]
                            then
                                python3.6 run.py --gamma $gamma -T 200000 --max_delay $d --tau $tau -k $k --fra_top $fra_top --delta $delta --delay_ub $ub --n_rep 5 -v 1 --bin $bin -s 1
                            else
                                python3.6 run.py --gamma $gamma -T 200000 --max_delay $d --tau $tau -k $k --fra_top $fra_top --delta $delta --delay_ub $ub --n_rep 1 -v 1 --bin $bin -s 1
                            fi
                        done
                    done
                done
            done
        done
    done
done
