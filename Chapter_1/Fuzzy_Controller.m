clc
clear all

array =[ 1 1 1 1 1;
1 2 1 1 1;
1 3 1 1 1;
1 4 1 1 1;
1 5 1 1 1;
1 6 2 1 1;
1 7 3 1 1;
1 8 4 1 1;
1 9 5 1 1;

2 1 1 1 1;
2 2 1 1 1;
2 3 2 1 1;
2 4 2 1 1;
2 5 2 1 1;
2 6 3 1 1;
2 7 4 1 1;
2 8 5 1 1;
2 9 6 1 1;

3 1 1 1 1;
3 2 2 1 1;
3 3 2 1 1;
3 4 2 1 1;
3 5 3 1 1;
3 6 4 1 1;
3 7 5 1 1;
3 8 6 1 1;
3 9 7 1 1;

4 1 2 1 1;
4 2 2 1 1;
4 3 2 1 1;
4 4 3 1 1;
4 5 4 1 1;
4 6 5 1 1;
4 7 6 1 1;
4 8 7 1 1;
4 9 8 1 1;

5 1 2 1 1;
5 2 2 1 1;
5 3 3 1 1;
5 4 4 1 1;
5 5 5 1 1;
5 6 6 1 1;
5 7 7 1 1;
5 8 8 1 1;
5 9 8 1 1;

6 1 2 1 1;
6 2 3 1 1;
6 3 4 1 1;
6 4 5 1 1;
6 5 6 1 1;
6 6 7 1 1;
6 7 8 1 1;
6 8 8 1 1;
6 9 8 1 1;

7 1 3 1 1;
7 2 4 1 1;
7 3 5 1 1;
7 4 6 1 1;
7 5 7 1 1;
7 6 8 1 1;
7 7 8 1 1;
7 8 8 1 1;
7 9 9 1 1;

8 1 4 1 1;
8 2 5 1 1;
8 3 6 1 1;
8 4 7 1 1;
8 5 8 1 1;
8 6 8 1 1;
8 7 8 1 1;
8 8 9 1 1;
8 9 9 1 1;

9 1 5 1 1;
9 2 6 1 1;
9 3 7 1 1;
9 4 8 1 1;
9 5 9 1 1;
9 6 9 1 1;
9 7 9 1 1;
9 8 9 1 1;
9 9 9 1 1];

Asafis_PI = readfis('Asafis_elegktis');
Asafis_PI.DefuzzificationMethod = "COS";
gensurf(Asafis_PI);

evalfis(Asafis_PI, [-0.5 0])
writeFIS(Asafis_PI,'Asafis_PI');

% Asafis_PI_2 =  readfis('Asafis_elegktis');
% Asafis_PI_2.DefuzzificationMethod = "centroid";
% fuzzyLogicDesigner(Asafis_PI_2);

ruleview(Asafis_PI)
