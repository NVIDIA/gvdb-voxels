// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include <iostream>
#include <fstream>
#include <string>
#include "sparse.h"

extern "C"
void readMatrixMarket(MMMatrix * m, const char * filename);

extern "C"
void sparseMatrixVectorMultiplyGold(const MMMatrix * m, const float * x, float * y);

using namespace std;

istream & operator>>(istream & is, MMEntry & mme)
{
    unsigned int row;
    unsigned int col;
    float entry;
    is >> row >> col >> entry;
    mme = MMEntry(row-1, col-1, entry); // convert to 0-based
    return is;
}

ostream & operator<<(ostream & os, const MMEntry & mme)
{
    os << mme.getRow() << " " << mme.getCol() << " " << mme.getEntry() << endl;
    return os;
}

istream & operator>>(istream & is, MMMatrix & m)
{
    unsigned int rows;
    unsigned int cols;
    unsigned int count;
    is >> rows >> cols >> count;
    m = MMMatrix(rows, cols, count);
    return is;
}

ostream & operator<<(ostream & os, const MMMatrix & m)
{
    os << m.getRows() << " " << m.getCols() << " " << m.getNumEntries() << endl;
    unsigned int i;
    for (i = 0; i < m.getNumEntries(); i++)
    {
        os << m[i];
    }
    for (i = 0; i < m.getRows(); i++)
    {
        os << m.getRowPtr(i) << ' ';
    }
    os << endl;
    return os;
}

int mmecmp(const void * aa, const void * bb) {
    MMEntry * a = (MMEntry *) aa;
    MMEntry * b = (MMEntry *) bb;
    if (a->getRow() < b->getRow())
    {
        return -1;
    }
    if (a->getRow() > b->getRow())
    {
        return 1;
    }
    if (a->getCol() < b->getCol())
    {
        return -1;
    }
    if (a->getCol() > b->getCol())
    {
        return 1;
    }
    return 0;
}


void readMatrixMarket(MMMatrix * m, const char * filename)
{
    ifstream f(filename);
    if (!f)
    {
        cerr << "Cannot open file " << filename << endl;
        exit(1);
    }

    string s;
    char c = 0;
    while ((c = f.get()) && (c != EOF))
    {
        f.putback(c);
        if (c != '%') 
        {
            break;
        }
        getline(f, s);
    }

    // now we've cleared all comments away
    f >> *m;
    
    MMEntry e;
    unsigned int i;
    for (i = 0 ; i < m->getNumEntries() ; i++)
    {
        f >> e;
        m->setEntry(i, e);
    }
    
    // sort into row-major order
    qsort(m->getEntriesForSorting(), m->getNumEntries(), 
          sizeof(MMEntry), mmecmp);
    
    // set index of first elt in each row
    // relies on at least one item in each row
    unsigned int row;
    for (i = 0 ; i < m->getNumEntries() ; i++)
    {
        row = (*m)[i].getRow();
        if (m->getRowPtr(row) > i)
        {
            m->setRowPtr(row, i);
        }
    }
    
    // now set rowFPtr
    for (row = 0 ; row < m->getRows() - 1 ; row++)
    {
        m->setRowFPtr(row, m->getRowPtr(row+1) - 1);
    }
    m->setRowFPtr(m->getRows() - 1, m->getNumEntries() - 1);
}

void sparseMatrixVectorMultiplyGold(const MMMatrix * m, const float * x, float * y)
{
    // unsigned int i = 0;
    // for (i = 0; i < m->getCols(); i++)
    // {
    //     y[i] = 0.0f;
    // }
    for (unsigned int i = 0; i < m->getNumEntries(); i++)
    {
        MMEntry e((*m)[i]);
        y[e.getRow()] += e.getEntry() * x[e.getCol()];
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
