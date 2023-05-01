#include <iostream>
#include "TemplatedVocabulary.h"
#include "DBoW2.h"

int main()
{

	typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

	std::string filename = "../../../loftr_voc_og.txt";

	// saveToTextFile(filename);

	std::cout << "Done!\n";
}
