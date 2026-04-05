/* seir_multiseed.c — SEIR 730 jours, seed variable, pour spaghetti plot */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define GRID_W 300
#define GRID_H 300
#define N_AGENTS 20000
#define N_INIT_I 20
#define N_STEPS 730
#define BETA 0.5
#define MEAN_E 3.0
#define MEAN_I 7.0
#define MEAN_R 365.0
#define FRAME_EVERY 5
#define ST_S 0
#define ST_E 1
#define ST_I 2
#define ST_R 3
typedef struct { int status,time_in_state,dE,dI,dR,x,y; } Agent;
static Agent agents[N_AGENTS];
static int cell_head[GRID_H*GRID_W], cell_next[N_AGENTS];
static int new_status[N_AGENTS], new_time[N_AGENTS];
static inline double rand01(void){return rand()/(double)RAND_MAX;}
static inline int rand_int(int a,int b){return a+rand()%(b-a+1);}
static int negExp(double m){double u;do{u=rand01();}while(u>=1.0);int v=(int)ceil(-m*log(1.0-u));return v<1?1:v;}
static void init_agents(void){for(int i=0;i<N_AGENTS;i++){agents[i].status=(i<N_INIT_I)?ST_I:ST_S;agents[i].time_in_state=0;agents[i].dE=negExp(MEAN_E);agents[i].dI=negExp(MEAN_I);agents[i].dR=negExp(MEAN_R);agents[i].x=rand_int(0,GRID_W-1);agents[i].y=rand_int(0,GRID_H-1);}}
static void rebuild_grid(void){memset(cell_head,-1,sizeof(cell_head));for(int i=0;i<N_AGENTS;i++){int idx=agents[i].y*GRID_W+agents[i].x;cell_next[i]=cell_head[idx];cell_head[idx]=i;}}
static void step_move(void){for(int i=0;i<N_AGENTS;i++){agents[i].x=rand_int(0,GRID_W-1);agents[i].y=rand_int(0,GRID_H-1);}}
static int count_I_moore(int cx,int cy){int count=0;for(int dy=-1;dy<=1;dy++){int ny=((cy+dy)%GRID_H+GRID_H)%GRID_H;for(int dx=-1;dx<=1;dx++){int nx=((cx+dx)%GRID_W+GRID_W)%GRID_W;for(int ag=cell_head[ny*GRID_W+nx];ag!=-1;ag=cell_next[ag])if(agents[ag].status==ST_I)count++;}}return count;}
static void step_update(void){for(int i=0;i<N_AGENTS;i++){new_status[i]=agents[i].status;new_time[i]=agents[i].time_in_state+1;switch(agents[i].status){case ST_S:{int ni=count_I_moore(agents[i].x,agents[i].y);if(ni>0){double p=1.0-exp(-BETA*(double)ni);if(rand01()<p){new_status[i]=ST_E;new_time[i]=0;}}break;}case ST_E:if(new_time[i]>=agents[i].dE){new_status[i]=ST_I;new_time[i]=0;}break;case ST_I:if(new_time[i]>=agents[i].dI){new_status[i]=ST_R;new_time[i]=0;}break;case ST_R:if(new_time[i]>=agents[i].dR){new_status[i]=ST_S;new_time[i]=0;}break;}}for(int i=0;i<N_AGENTS;i++){agents[i].status=new_status[i];agents[i].time_in_state=new_time[i];}}
static void get_counts(int *s,int *e,int *ii,int *r){*s=*e=*ii=*r=0;for(int i=0;i<N_AGENTS;i++){switch(agents[i].status){case ST_S:(*s)++;break;case ST_E:(*e)++;break;case ST_I:(*ii)++;break;case ST_R:(*r)++;break;}}}
static void write_frame(FILE*fp){static int cnt[GRID_H*GRID_W][4];memset(cnt,0,sizeof(cnt));for(int i=0;i<N_AGENTS;i++)cnt[agents[i].y*GRID_W+agents[i].x][agents[i].status]++;unsigned char frame[GRID_H*GRID_W];for(int idx=0;idx<GRID_H*GRID_W;idx++){if(cnt[idx][ST_I])frame[idx]=3;else if(cnt[idx][ST_E])frame[idx]=2;else if(cnt[idx][ST_R])frame[idx]=4;else if(cnt[idx][ST_S])frame[idx]=1;else frame[idx]=0;}fwrite(frame,1,GRID_H*GRID_W,fp);}
int main(int argc,char**argv){
    int seed=(argc>1)?atoi(argv[1]):42;
    srand(seed);
    char fc[64],ff[64];
    snprintf(fc,sizeof(fc),"counts_seed_%d.csv",seed);
    snprintf(ff,sizeof(ff),"frames_seed_%d.bin",seed);
    FILE*f_counts=fopen(fc,"w"),*f_frames=fopen(ff,"wb");
    if(!f_counts||!f_frames){perror("fopen");return 1;}
    int hdr[2]={GRID_W,GRID_H};
    fwrite(hdr,sizeof(int),2,f_frames);
    fprintf(f_counts,"step,S,E,I,R\n");
    init_agents();rebuild_grid();
    int s,e,ii,r;get_counts(&s,&e,&ii,&r);
    fprintf(f_counts,"0,%d,%d,%d,%d\n",s,e,ii,r);
    write_frame(f_frames);
    for(int step=1;step<=N_STEPS;step++){step_move();rebuild_grid();step_update();get_counts(&s,&e,&ii,&r);fprintf(f_counts,"%d,%d,%d,%d,%d\n",step,s,e,ii,r);if(step%FRAME_EVERY==0)write_frame(f_frames);}
    printf("Seed %d OK -> %s\n",seed,fc);
    fclose(f_counts);fclose(f_frames);return 0;
}
