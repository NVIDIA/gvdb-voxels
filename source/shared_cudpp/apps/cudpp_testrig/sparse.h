
// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Revision: 3547 $
//  $Date: 2007-07-25 16:04:53 -0700 (Wed, 25 Jul 2007) $
//  -------------------------------------------------------------
//  
//  sparse.hpp
//  Sparse matrix support code (gold CPU version)
// 
// ***************************************************************

#ifndef __SPARSE_HPP__
#define __SPARSE_HPP__

#include <cstdlib>

/** Matrix entry in MatrixMarket format */
class MMEntry
{
private:
    unsigned int m_row;
    unsigned int m_col;
    float m_entry;
public:
    MMEntry(unsigned int row, unsigned int col, float entry) :
        m_row(row), m_col(col), m_entry(entry) {}
    MMEntry() : m_row(0), m_col(0), m_entry(0.0f) {}
    unsigned int getRow() const { return m_row; }
    unsigned int getCol() const { return m_col; }
    float getEntry() const { return m_entry; }
};

/** Matrix in MatrixMarket format */
class MMMatrix
{
private:
    unsigned int m_rows;
    unsigned int m_cols;
    unsigned int m_numEntries;
    MMEntry * m_entries;
    unsigned int * m_rowPtrs;
    unsigned int * m_rowFPtrs;
public:
    MMMatrix(unsigned int rows, unsigned int cols, unsigned int count) :
        m_rows(rows), m_cols(cols), m_numEntries(count)
        {
            m_entries = (MMEntry *) malloc(sizeof(MMEntry) * count);
            m_rowPtrs = (unsigned int *) malloc(sizeof(unsigned int) * rows);
            m_rowFPtrs = (unsigned int *) malloc(sizeof(unsigned int) * rows);
            for (unsigned int i = 0; i < rows; i++)
            {
                m_rowPtrs[i] = count + 1; // max
                m_rowFPtrs[i] = count + 1; // max
            }
        }
    MMMatrix() : m_rows(0), m_cols(0), m_entries(0) {}
    unsigned int getRows() const { return m_rows; }
    unsigned int getCols() const { return m_cols; }
    unsigned int getNumEntries() const { return m_numEntries; }
    const MMEntry & operator[](unsigned int idx) const
    {
        return m_entries[idx];
    }
    unsigned int getRowPtr(unsigned int idx) const
    {
        return m_rowPtrs[idx];
    }
    unsigned int getRowFPtr(unsigned int idx) const
    {
        return m_rowFPtrs[idx];
    }
    void setEntry(unsigned int idx, const MMEntry & e)
    {
        m_entries[idx] = e;
    }
    void setRowPtr(unsigned int idx, unsigned int val)
    {
        m_rowPtrs[idx] = val;
    }
    void setRowFPtr(unsigned int idx, unsigned int val)
    {
        m_rowFPtrs[idx] = val;
    }
    unsigned int * getRowPtrs() const
    {
        return m_rowPtrs;
    }
    unsigned int * getRowFPtrs() const
    {
        return m_rowFPtrs;
    }
    MMEntry * getEntriesForSorting()
    {
        return m_entries;
    }
    ~MMMatrix()
    {
        // free(m_entries);
        // free(m_rowPtrs);
        // free(m_rowPtrFs);
        // clearly we should call this, but it has a compile error (!)
    }
};

void computeSaGold(unsigned char* idata, unsigned int* reference, size_t len);

#endif /* __SPARSE_HPP__ */

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
