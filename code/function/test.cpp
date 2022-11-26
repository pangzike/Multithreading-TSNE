#include<iostream>

using namespace std;
int main(int argc, char const *argv[])
{
	int height = 10000, tid = 0;
	int diag = (tid / height == tid % height)? 1:0;
	cout<< diag<<endl;
	return 0;
}
