#!/bin/bash 

set -e 

file_name=${1}

mkdir -p  01_min 02_min 03_em_npt


##################################### MINI-RES + MINI + EM (STANDARD W)  
cd 01_min
cp ../files/*.pdb .
cp ../files/*.itp .
cp ../files/*.top .

#### change TOP
sed -i "s/ZZ/W/g" top.top
sed -i "5 s/;//g" top.top

cp ../files/*.mdp .

gmx  grompp -c ${file_name} -r ${file_name} -p top.top -f min-res-h.mdp -o min_01.tpr
gmx  mdrun -deffnm min_01 -ntmpi 2 -ntomp 12 & 
pid=$!
wait ${pid}

gmx  grompp -c min_01.gro  -t min_01.trr -p top.top -f min-h.mdp -o min_02.tpr
gmx  mdrun -deffnm min_02 -ntmpi 2 -ntomp 12 & 
pid=$!
wait ${pid}


gmx grompp -c min_02.gro -r min_02.gro -t min_02.trr -p top.top -f em-nvt-res-h.mdp -o min_03.tpr
gmx mdrun -deffnm min_03 -ntmpi 2 -ntomp 12 &
pid=$!
wait ${pid}

gmx grompp -c min_03.gro -p top.top -f em-nvt-h.mdp -o min_04.tpr
gmx mdrun -deffnm min_04 -ntmpi 2 -ntomp 12 &
pid=$!
wait ${pid}


cp ../files/triple-w.py .
python2 triple-w.py min_04.gro 

cd ../




###################################### RES MINI
cd 02_min
cp ../01_min/min_04_PW.gro .
cp ../files/*.itp .
cp ../files/*.top .
cp ../files/*.mdp .

### change TOP
sed -i "s/ZZ/PW/g" top.top
sed -i "6 s/;//g" top.top

cp ../files/*.mdp .
cp ../files/posre.itp .

gmx grompp -c min_04_PW.gro  -r min_04_PW.gro -p top.top -f min-res-h.mdp -o min_01.tpr
gmx mdrun -deffnm min_01 -ntmpi 2 -ntomp 12 &
pid=$!
wait ${pid}


cp ../files/top.top .
sed -i "s/ZZ/PW/g" top.top
sed -i "7 s/;//g" top.top

gmx grompp -c min_01.gro  -p top.top -f em-nvt-h.mdp -o min_02.tpr
gmx mdrun -deffnm min_02 -ntmpi 2 -ntomp 12 &
pid=$!
wait ${pid}

cd ..



##################################### EM
cd 03_em_npt
cp ../02_min/min_02.gro .
cp ../files/*.itp .
cp ../files/*.top .
cp ../files/*.mdp .

### change TOP
sed -i "s/ZZ/PW/g" top.top
sed -i "7 s/;//g" top.top

gmx grompp -c min_02.gro  -p top.top -f em-npt-h.mdp -o em.tpr
#gmx mdrun -deffnm em -ntmpi 2 -ntomp 12 &


cd ..








