#!/bin/bash

correct=0
total=0
scalability=0
correctness=0
correct_scalability=0
times=()
seq_times=()
par_times=()
seq_runs=0

# afiseaza scorul final
function show_score {
    echo ""
    echo "Scalabilitate: $scalability/64"
    echo "Corectitudine: $correctness/56"
    echo "Total:       $((correctness + scalability))/120"
}

# se compara doua fisiere (parametri: fisier1 fisier2)
function compare_files {
    ok=0

    diff -q $1 $2
    if [ $? == 0 ]
    then
        correct=$((correct+1))
    else
        echo "W: Exista diferente intre fisierele $1 si $2"
        correct_scalability=$((correct_scalability-1))
    fi
}

# se ruleaza o comanda si se masoara timpul (parametru: comanda)
function run_and_get_time {
    { time -p sh -c "timeout 10 $1"; } &> time.txt
    ret=$?

    if [ $ret == 124 ]
    then
        echo "E: Programul a durat mai mult de 10 s"
        cat time.txt | sed '$d' | sed '$d' | sed '$d'
        show_score
        rm -rf test*_sec.ppm
        exit
    elif [ $ret != 0 ]
    then
        echo "E: Rularea nu s-a putut executa cu succes"
        cat time.txt | sed '$d' | sed '$d' | sed '$d'
        show_score
        rm -rf test*_sec.ppm
        exit
    fi

    res=`cat time.txt | grep real | awk '{print $2}'`
    seq_times+=(${res})
    times+=(${res%.*})

    rm -rf time.txt

    seq_runs=$((seq_runs+1))
}

# se ruleaza si masoara o comanda paralela (parametri: timeout comanda)
function run_par_and_measure {
    { time -p sh -c "timeout $1 $2" ; } &> time.txt
    ret=$?

    if [ $ret == 124 ]
    then
        echo "W: Programul a durat cu cel putin 2 secunde in plus fata de implementarea secventiala"
    elif [ $ret != 0 ]
    then
        echo "W: Rularea nu s-a putut executa cu succes"
        cat time.txt | sed '$d' | sed '$d' | sed '$d'
    fi

    total=$((total+1))
    par_times+=(`cat time.txt | grep real | awk '{print $2}'`)

    rm -rf time.txt
}

# se ruleaza o comanda paralela fara sa se masoare (parametri: timeout comanda)
function run_par {
    { time -p sh -c "timeout $1 $2" ; } &> time.txt
    ret=$?

    if [ $ret == 124 ]
    then
        echo "W: Programul a durat cu cel putin 2 secunde in plus fata de implementarea secventiala"
    elif [ $ret != 0 ]
    then
        echo "W: Rularea nu s-a putut executa cu succes"
        cat time.txt | sed '$d' | sed '$d' | sed '$d'
    fi

    total=$((total+1))

    rm -rf time.txt
}

echo "VMCHECKER_TRACE_CLEANUP"
date

# se compileaza tema
cd ../src
ls -lh
rm -rf tema1_par
cp ../checker/helpers.* .
make clean &> /dev/null
make build &> build.txt

if [ ! -f tema1_par ]
then
    echo "E: Nu s-a putut compila tema"
    cat build.txt
    show_score
    rm -rf build.txt
    exit
fi

rm -rf build.txt

mv tema1_par ../checker
cd ../checker
make build &> /dev/null

for i in `seq 1 7`
do
    echo ""
    echo "======== Testul ${i} ========"
    echo ""
    echo "Se ruleaza varianta secventiala..."

    run_and_get_time "./tema1 inputs/in_${i}.ppm test${i}_sec.ppm"

    echo "Rularea a durat ${seq_times[$i - 1]} secunde"

    correct_scalability=6

    echo "OK"
    echo ""

    # se creste valoarea de timeout cu 2 secunde
    times[$i]=$((times[$i]+2))

    # se ruleaza implementarea paralela pe 2 si 4 thread-uri
    for P in 2 4
    do
        echo "Se ruleaza varianta cu $P thread-uri..."

        # se ruleaza de doua ori aditionale, pentru validarea rezultatului
        for j in `seq 1 2`
        do
            run_par ${times[$i]} "./tema1_par inputs/in_${i}.ppm test${i}_par.ppm $P"
            compare_files test${i}_sec.ppm test${i}_par.ppm
        done

        # se masoara timpii paraleli (pentru calculul acceleratiei)
        run_par_and_measure ${times[$i]} "./tema1_par inputs/in_${i}.ppm test${i}_par.ppm $P"

        if [ $P == 2 ]
        then
            echo "Rularea a durat ${par_times[$i * 2 - 2]} secunde"
        else
            echo "Rularea a durat ${par_times[$i * 2 - 1]} secunde"
        fi

        # se tine minte daca rezultatul e corect (pentru punctajul de scalabilitate)
        compare_files test${i}_sec.ppm test${i}_par.ppm

        echo "OK"
        echo ""
    done

    rm -rf test${i}_seq.ppm test${i}_par.ppm

    if [ $i == 6 ] || [ $i == 7 ]
    then
        # se calculeaza acceleratia
        speedup12=$(echo "${seq_times[$i - 1]}/${par_times[$i * 2 - 2]}" | bc -l | xargs printf "%.2f")
        speedup14=$(echo "${seq_times[$i - 1]}/${par_times[$i * 2 - 1]}" | bc -l | xargs printf "%.2f")

        # acceleratia se considera 0 daca testele de scalabilitate nu sunt corecte
        if [ $correct_scalability != 6 ]
        then
            speedup12=0
            speedup14=0
            echo "Testele nu dau rezultate corecte, acceleratia se considera 0"
        fi

        printf "Acceleratie 1-2 Mappers: %0.2f\n" $speedup12
        printf "Acceleratie 1-4 Mappers: %0.2f\n" $speedup14

        if [ $i != 1 ]
        then
            # se verifica acceleratia de la secvential la 2 thread-uri
            max=$(echo "${speedup12} > 1.6" | bc -l)
            part=$(echo "${speedup12} > 1.3" | bc -l)
            if [ $max == 1 ]
            then
                scalability=$((scalability+16))
            elif [ $part == 1 ]
            then
                scalability=$((scalability+8))
                echo "W: Acceleratia de la 1 la 2 Mappers este prea mica (punctaj partial)"
            else
                echo "W: Acceleratia de la 1 la 2 Mappers este prea mica (fara punctaj)"
            fi
        fi

        # se verifica acceleratia de la secvential la 4 thread-uri
        max=$(echo "${speedup14} > 2.6" | bc -l)
        part=$(echo "${speedup14} > 2.3" | bc -l)
        if [ $max == 1 ]
        then
            scalability=$((scalability+16))
        elif [ $part == 1 ]
        then
            scalability=$((scalability+8))
            echo "W: Acceleratia de la 1 la 4 Mappers este prea mica (punctaj partial)"
        else
            echo "W: Acceleratia de la 1 la 4 Mappers este prea mica (fara punctaj)"
        fi
    fi

    echo "=========================="
    echo ""
done

# punctajul pe corectitudine este conditionat de minim un test de scalabilitate trecut
if [ $scalability == 0 ]
then
    echo "W: Au picat toate testele de scalabilitate, corectitudinea se considera 0"
    correctness=0
else
    correctness=$((correct * 56 / total))
fi

cd ../src
make clean &> /dev/null

show_score
