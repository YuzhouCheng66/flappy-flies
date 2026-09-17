// Full graph + full recurrent readout. SIMD changes reduction/activation rounding,
// not the graph, checkpoint, agent count, or four updates per physics tick.
#include <emscripten/emscripten.h>
#include <wasm_simd128.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int N,threads,job;
static uint32_t *rows,*edges,*inputs,*outputs,*display,*offsets;
static float *denom,*weights,*state,*next_state,*obs;
static float drive[2048*8],hidden[8*128],result[6208];
static int boundaries[17];
static pthread_t workers[16];
static pthread_barrier_t start_barrier,end_barrier;
static float *w(int k){return weights+offsets[k];}
static float clampf(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
static v128_t splat(float x){return wasm_f32x4_splat(x);}
// exp(-2|x|) using exact powers of two and a degree-8 Taylor polynomial
// on [-ln(2)/2,ln(2)/2]. Numerical equivalence is tested, not assumed.
static v128_t fast_tanh(v128_t x){
 v128_t a=wasm_f32x4_min(wasm_f32x4_abs(x),splat(9));
 v128_t y=wasm_f32x4_mul(a,splat(-2));
 v128_t nf=wasm_f32x4_nearest(wasm_f32x4_mul(y,splat(1.4426950408889634f)));
 v128_t r=wasm_f32x4_sub(y,wasm_f32x4_mul(nf,splat(.6931471805599453f)));
 v128_t p=splat(1.f/40320);
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1.f/5040));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1.f/720));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1.f/120));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1.f/24));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1.f/6));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(.5));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1));
 p=wasm_f32x4_add(wasm_f32x4_mul(p,r),splat(1));
 v128_t exponent=wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(nf),wasm_i32x4_splat(127)),23);
 v128_t e=wasm_f32x4_mul(p,exponent);
 v128_t value=wasm_f32x4_div(wasm_f32x4_sub(splat(1),e),wasm_f32x4_add(splat(1),e));
 return wasm_v128_xor(value,wasm_v128_and(x,wasm_i32x4_splat((int32_t)0x80000000)));
}
static float dot(const float *a,const float *b,int n){
 v128_t sum=splat(0);int j=0;
 for(;j+3<n;j+=4)sum=wasm_f32x4_add(sum,wasm_f32x4_mul(wasm_v128_load(a+j),wasm_v128_load(b+j)));
 float value=wasm_f32x4_extract_lane(sum,0)+wasm_f32x4_extract_lane(sum,1)+wasm_f32x4_extract_lane(sum,2)+wasm_f32x4_extract_lane(sum,3);
 for(;j<n;j++)value+=a[j]*b[j];return value;
}
static void sensory(int a){
 float normalized[27];for(int j=0;j<27;j++)normalized[j]=clampf((obs[a*27+j]-w(20)[j])/w(21)[j],-8,8);
 for(int r=0;r<2048;r++)drive[r*8+a]=w(19)[r]+dot(w(18)+r*27,normalized,27);
}
static void recurrent(int lo,int hi){
 for(int r=lo;r<hi;r++){
  v128_t a=splat(0),b=splat(0);
  for(uint32_t j=rows[r];j<rows[r+1];j++){
   uint32_t edge=edges[j],c=(edge&262143)*8;v128_t v=splat((float)(edge>>18));
   a=wasm_f32x4_add(a,wasm_f32x4_mul(v,wasm_v128_load(state+c)));
   b=wasm_f32x4_add(b,wasm_f32x4_mul(v,wasm_v128_load(state+c+4)));
  }
  v128_t scale=splat(.5f/denom[r]);a=wasm_f32x4_mul(a,scale);b=wasm_f32x4_mul(b,scale);
  uint32_t id=inputs[r];if(id!=UINT32_MAX){a=wasm_f32x4_add(a,wasm_v128_load(drive+id*8));b=wasm_f32x4_add(b,wasm_v128_load(drive+id*8+4));}
  wasm_v128_store(next_state+r*8,wasm_f32x4_mul(splat(.5),wasm_f32x4_add(wasm_v128_load(state+r*8),fast_tanh(a))));
  wasm_v128_store(next_state+r*8+4,wasm_f32x4_mul(splat(.5),wasm_f32x4_add(wasm_v128_load(state+r*8+4),fast_tanh(b))));
 }
}
static void dense(const float *x,float *y,int in,int out,int wi,int bi,int activation){
 for(int r=0;r<out;r++){float s=w(bi)[r]+dot(w(wi)+r*in,x,in);y[r]=activation?s/(1+expf(-s)):s;}
}
static void decode(int a){
 float z[512],l1[256],l2[256],ff[6],embed[128],updated[128],residual[6];
 for(int j=0;j<512;j++){float value=state[outputs[j]*8+a];result[2104+a*512+j]=value;z[j]=clampf((value-w(0)[j])/w(1)[j],-10,10);}
 dense(z,l1,512,256,4,5,1);dense(l1,l2,256,256,6,7,1);dense(l2,ff,256,6,8,9,0);dense(z,embed,512,128,10,11,1);
 for(int u=0;u<128;u++){
  float iv[3],hv[3];for(int g=0;g<3;g++){int r=u+g*128;iv[g]=w(14)[r]+dot(w(12)+r*128,embed,128);hv[g]=w(15)[r]+dot(w(13)+r*128,hidden+a*128,128);}
  float reset=1/(1+expf(-iv[0]-hv[0])),update=1/(1+expf(-iv[1]-hv[1]));
  updated[u]=(1-update)*tanhf(iv[2]+reset*hv[2])+update*hidden[a*128+u];
 }
 memcpy(hidden+a*128,updated,sizeof(updated));dense(updated,residual,128,6,16,17,0);
 for(int j=0;j<6;j++)result[a*6+j]=(ff[j]+residual[j])*w(3)[j]+w(2)[j];
 double sum=0;for(int r=0;r<N;r++){float v=state[r*8+a];sum+=(double)v*v;}result[2096+a]=sqrt(sum/N);
}
static void *worker(void *arg){int id=(int)(intptr_t)arg;for(;;){
 pthread_barrier_wait(&start_barrier);
 if(job==1)for(int a=id;a<8;a+=threads)sensory(a);
 if(job==2)recurrent(boundaries[id],boundaries[id+1]);
 if(job==3)for(int a=id;a<8;a+=threads)decode(a);
 pthread_barrier_wait(&end_barrier);
}return NULL;}
static void run(int mode){job=mode;pthread_barrier_wait(&start_barrier);pthread_barrier_wait(&end_barrier);}
EMSCRIPTEN_KEEPALIVE int setup(int n,uint32_t *r,uint32_t *e,float *d,uint32_t *i,uint32_t *o,uint32_t *v,float *weight,uint32_t *off,int nt){
 if(nt<1||nt>16)return -1;N=n;rows=r;edges=e;denom=d;inputs=i;outputs=o;display=v;weights=weight;offsets=off;threads=nt;
 state=calloc(N*8,sizeof(float));next_state=calloc(N*8,sizeof(float));if(!state||!next_state)return -2;
 boundaries[0]=0;boundaries[threads]=N;int k=1;for(int row=0;row<N&&k<threads;row++)while(k<threads&&rows[row]>=(uint64_t)rows[N]*k/threads)boundaries[k++]=row;
 pthread_barrier_init(&start_barrier,NULL,threads+1);pthread_barrier_init(&end_barrier,NULL,threads+1);
 for(int t=0;t<threads;t++)if(pthread_create(workers+t,NULL,worker,(void*)(intptr_t)t))return -3;return 0;
}
EMSCRIPTEN_KEEPALIVE void reset_model(void){memset(state,0,N*8*sizeof(float));memset(next_state,0,N*8*sizeof(float));memset(hidden,0,sizeof(hidden));}
EMSCRIPTEN_KEEPALIVE float *evaluate(float *observation){obs=observation;double start=emscripten_get_now();run(1);result[6200]=emscripten_get_now()-start;start=emscripten_get_now();for(int t=0;t<4;t++){run(2);float *tmp=state;state=next_state;next_state=tmp;}result[6201]=emscripten_get_now()-start;start=emscripten_get_now();run(3);result[6202]=emscripten_get_now()-start;for(int j=0;j<2048;j++)result[48+j]=state[display[j]*8];return result;}
