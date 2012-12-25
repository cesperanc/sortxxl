 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 #include <time.h>
 
 
int genRandom (unsigned long numElms, long int *vectElms){
	long int Num;
	unsigned long i;
	 
	 for(i=0; i< numElms; i++){
		Num = random();
		vectElms[i]=Num;
	 }
	 
	 return i;
}
 
int genRandomMinMax(unsigned long numElms, long int *vectElms, long int min, long int max){
	 
	 long int Num;
	 unsigned long i;
	 unsigned long Diff;
	 
	 assert(vectElms != NULL);
	 assert(min<= max);
	 
	 Diff = max-min+1;
	 for(i=0; i< numElms; i++){
		Num = (random() %Diff)+min;
		vectElms[i]=Num;
		/*DEBUG*/
		printf("[%lu]=%lu\n",i,Num);
		assert(Num>= min);
		assert(Num<= max);
	 }
	 
	 return i;
}
/*seed the random generator*/
void InitRandom(void){
		srandom(time(NULL));
}

void showVect(long *vectPtr, unsigned long numElms){
	unsigned long i;
	assert(vectPtr != NULL);
	
	for(i=0; i< numElms; i++){
		printf("[%lu]=%lu\n",i,vectPtr[i]);
	}
	
}
int main(){
	
	long int  NumElms = 1E4;
	long int NumRandom;
	long int RandomVectSize = NumElms*sizeof(long int);
	long int *RandomElms_L;
	
	InitRandom();
	
	RandomElms_L = malloc(RandomVectSize);
	
	if(RandomElms_L == NULL){
		fprintf(stderr, "[ERROR] can alloc %ld bytes\n", RandomVectSize);
	}
	
	
	//NumRandom = genRandom(NumElms, RandomElms_L);
	long int Min = 150;
	long int Max = 99999;
	NumRandom = genRandomMinMax(NumElms, RandomElms_L, Min, Max);
	assert(NumRandom == NumElms);
	
	showVect(RandomElms_L, NumElms );
	free(RandomElms_L);
	
	return 0;
}
