import re
from itertools import product
from collections import Counter


def blacklist_missing(loci, threshold, iupac=False):
    """
    Identifies loci with a proportion of missing data above a given threshold.

    This function takes a list of loci and a threshold, and returns a list of indices for loci where the proportion of missing data is above the threshold. If `iupac` is True, it treats the genotypes as IUPAC codes.

    Args:
        loci (List[List[int or str]]): The list of loci to be checked. Each locus is a list of genotypes encoded as 0, 1, or 2.

        threshold (float): The missing data threshold. Loci with a proportion of missing data above this threshold will be included in the blacklist.

        iupac (bool, optional): If True, the genotypes are treated as IUPAC codes. Defaults to False.

    Returns:
        List[int]: A list of indices for loci where the proportion of missing data is above the threshold.

    Example:
        loci = [[0, 1, -9], [0, 0, 0], [1, -9, -9]]
        threshold = 0.5
        print(blacklist_missing(loci, threshold))  # Outputs: [2]

    Note:
        The function uses the ``expandLoci`` function to expand each locus into a list of alleles, and the ``collections.Counter`` class to count the occurrences of each allele. Missing data is represented by -9.
    """
    blacklist = list()
    for i in range(0, len(loci)):
        alleles = expandLoci(loci[i], iupac=False)
        c = Counter(alleles)
        if float(c[-9] / sum(c.values())) > threshold:
            blacklist.append(i)
    return blacklist


def blacklist_maf(loci, threshold, iupac=False):
    """
    Identifies loci with minor allele frequency (MAF) below a given threshold.

    This function takes a list of loci and a threshold, and returns a list of indices for loci where the MAF is below the threshold. If `iupac` is True, it treats the genotypes as IUPAC codes.

    Args:
        loci (List[List[int or str]]): The list of loci to be checked. Each locus is a list of genotypes encoded as 0, 1, or 2.

        threshold (float): The MAF threshold. Loci with MAF below this threshold will be included in the blacklist.

        iupac (bool, optional): If True, the genotypes are treated as IUPAC codes. Defaults to False.

    Returns:
        List[int]: A list of indices for loci where the MAF is below the threshold.

    Example:
        loci = [[0, 1, 2], [0, 0, 0], [1, 1, 2]]
        threshold = 0.2
        print(blacklist_maf(loci, threshold))  # Outputs: [1]

    Note:
        The function uses the `expandLoci` function to expand each locus into a list of alleles, and the `collections.Counter` class to count the occurrences of each allele.
    """
    blacklist = list()
    for i in range(0, len(loci)):
        alleles = expandLoci(loci[i], iupac=False)
        c = Counter(alleles)
        if len(c.keys()) <= 1:
            blacklist.append(i)
            continue
        else:
            minor_count = c.most_common(2)[1][1]
            if float(minor_count / sum(c.values())) < threshold:
                blacklist.append(i)
    return blacklist


def expandLoci(loc, iupac=False):
    """
    Expands a list of genotypes encoded as 0, 1, or 2 into a list of alleles.

    This function takes a list of genotypes encoded as 0 (homozygous reference), 1 (heterozygous), or 2 (homozygous alternate) and returns a list of alleles corresponding to the genotypes. If `iupac` is True, it treats the genotypes as IUPAC codes and expands them accordingly.

    Args:
        loc (List[int or str]): The list of genotypes to be expanded, each encoded as 0, 1, or 2.

        iupac (bool, optional): If True, the genotypes are treated as IUPAC codes. Defaults to False.

    Returns:
        List[int]: A list of alleles corresponding to the input genotypes.

    Example:
        loc = [0, 1, 2]
        print(expandLoci(loc))  # Outputs: [0, 0, 0, 1, 1, 1]

    Note:
        The function uses the ``expand012`` function to expand each genotype. If ``iupac`` is True, it uses the ``get_iupac_caseless`` function instead.
    """
    ret = list()
    for i in loc:
        if not iupac:
            ret.extend(expand012(i))
        else:
            ret.extent(get_iupac_caseless(i))
    return ret


def expand012(geno):
    """
    Expands a genotype encoded as 0, 1, or 2 into a list of two alleles.

    This function takes a genotype encoded as 0 (homozygous reference), 1 (heterozygous), or 2 (homozygous alternate) and returns a list of two alleles corresponding to the genotype. Any other input returns a list of two missing data values (-9).

    Args:
        geno (int or str): The genotype to be expanded, encoded as 0, 1, or 2.

    Returns:
        List[int]: A list of two alleles corresponding to the input genotype.

    Example:
        geno = 1
        print(expand012(geno))  # Outputs: [0, 1]

    Note:
        The function assumes that the input genotype is one of 0, 1, or 2. Any other input will be treated as missing data.
    """
    g = str(geno)
    if g == "0":
        return [0, 0]
    elif g == "1":
        return [0, 1]
    elif g == "2":
        return [1, 1]
    else:
        return [-9, -9]


def remove_items(all_list, bad_list):
    """
    Removes items from a list based on another list.

    This function takes a list and removes any items that are present in a second list.

    Args:
        all_list (List[Any]): The list from which items are to be removed.

        bad_list (List[Any]): The list containing items to be removed from the first list.

    Returns:
        List[Any]: The first list with any items present in the second list removed.

    Example:
        all_list = ['a', 'b', 'c', 'd']
        bad_list = ['b', 'd']
        print(remove_items(all_list, bad_list))  # Outputs: ['a', 'c']
    """  # using list comprehension to perform the task
    return [i for i in all_list if i not in bad_list]


def count_alleles(l, vcf=False):
    """
    Counts the total number of unique alleles in a list of genotypes.

    This function takes a list of IUPAC or VCF-style (e.g. 0/1) genotypes and returns the total number of unique alleles. The genotypes can be in VCF or STRUCTURE-style format.

    Args:
        l (List[str]): A list of IUPAC or VCF-style genotypes.

        vcf (bool, optional): If True, the genotypes are in VCF format. If False, the genotypes are in STRUCTURE-style format. Defaults to False.

    Returns:
        int: The total number of unique alleles in the list.

    Example:
        l = ['A/A', 'A/T', 'T/T', 'A/A', 'A/T']
        print(count_alleles(l, vcf=True))  # Outputs: 2

    Note:
        The function removes any instances of "-9", "-", "N", -9, ".", "?" before counting the alleles.
    """
    all_items = list()
    for i in l:
        if vcf:
            all_items.extend(i.split("/"))
        else:
            all_items.extend(get_iupac_caseless(i))
    all_items = remove_items(all_items, ["-9", "-", "N", -9, ".", "?"])
    return len(set(all_items))


def get_major_allele(l, num=None, vcf=False):
    """
    Returns the most common alleles in a list.

    This function takes a list of genotypes for one sample and returns the most common alleles in descending order. The alleles can be in VCF or STRUCTURE-style format.

    Args:
        l (List[str]): A list of genotypes for one sample.

        num (int, optional): The number of elements to return. If None, all elements are returned. Defaults to None.

        vcf (bool, optional): If True, the alleles are in VCF format. If False, the alleles are in STRUCTURE-style format. Defaults to False.

    Returns:
        List[str]: The most common alleles in descending order.

    Example:
        l = ['A/A', 'A/T', 'T/T', 'A/A', 'A/T']
        print(get_major_allele(l, vcf=True))  # Outputs: ['A', 'T']

    Note:
        The function uses the Counter class from the collections module to count the occurrences of each allele.
    """
    all_items = list()
    for i in l:
        if vcf:
            all_items.extend(i.split("/"))
        else:
            all_items.extend(get_iupac_caseless(i))

    c = Counter(all_items)  # requires collections import

    # List of tuples with [(allele, count), ...] in order of
    # most to least common
    rets = c.most_common(num)

    # Returns two most common non-ambiguous bases
    # Makes sure the least common base isn't N or -9
    if vcf:
        return [x[0] for x in rets if x[0] != "-9"]
    else:
        return [x[0] for x in rets if x[0] in ["A", "T", "G", "C"]]


def get_iupac_caseless(char):
    """Split IUPAC code to two primary characters, assuming diploidy.

    Gives all non-valid ambiguities as N.

    Args:
        char (str): Base to expand into diploid list.

    Returns:
        List[str]: List of the two expanded alleles.
    """
    lower = False
    if char.islower():
        lower = True
        char = char.upper()
    iupac = {
        "A": ["A", "A"],
        "G": ["G", "G"],
        "C": ["C", "C"],
        "T": ["T", "T"],
        "N": ["N", "N"],
        "-": ["N", "N"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "S": ["G", "C"],
        "W": ["A", "T"],
        "K": ["G", "T"],
        "M": ["A", "C"],
        "B": ["N", "N"],
        "D": ["N", "N"],
        "H": ["N", "N"],
        "V": ["N", "N"],
        "-9": ["N", "N"],
    }
    ret = iupac[char]
    if lower:
        ret = [c.lower() for c in ret]
    return ret


def get_iupac_full(char):
    """Split IUPAC code to all possible primary characters.

    Gives all ambiguities as "N".

    Args:
        char (str): Base to expaned into list.

    Returns:
        List[str]: List of the expanded alleles.
    """
    char = char.upper()
    iupac = {
        "A": ["A"],
        "G": ["G"],
        "C": ["C"],
        "T": ["T"],
        "N": ["A", "C", "T", "G"],
        "-": ["A", "C", "T", "G"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "S": ["G", "C"],
        "W": ["A", "T"],
        "K": ["G", "T"],
        "M": ["A", "C"],
        "B": ["C", "G", "T"],
        "D": ["A", "G", "T"],
        "H": ["A", "C", "T"],
        "V": ["A", "C", "G"],
    }
    ret = iupac[char]
    return ret


def expandAmbiquousDNA(sequence):
    """Generator function to expand ambiguous sequences"""
    for i in product(*[get_iupac_caseless(j) for j in sequence]):
        yield ("".join(i))


def get_revComp_caseless(char):
    """
    Returns the reverse complement of a nucleotide, while preserving case.

    This function takes a nucleotide character and returns its reverse complement according to the standard DNA base pairing rules. It also handles IUPAC ambiguity codes. The case of the input character is preserved in the output.

    Args:
        char (str): The nucleotide character to be reverse complemented. Can be uppercase or lowercase.

    Returns:
        str: The reverse complement of the input character, with the same case.

    Example:
        char = 'a'
        print(get_revComp_caseless(char))  # Outputs: 't'

    Note:
        The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G). It also supports N (any base) and - (gap).
    """
    lower = False
    if char.islower():
        lower = True
        char = char.upper()
    d = {
        "A": "T",
        "G": "C",
        "C": "G",
        "T": "A",
        "N": "N",
        "-": "-",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "D": "H",
        "H": "D",
        "V": "B",
    }
    ret = d[char]
    if lower:
        ret = ret.lower()
    return ret


def reverseComplement(seq):
    """
    Returns the reverse complement of a DNA sequence while preserving case.

    This function takes a DNA sequence and returns its reverse complement. The function preserves the case of the input sequence. For example, if the input sequence contains uppercase letters, the output will also contain uppercase letters, and vice versa for lowercase letters.

    Args:
        seq (str): The DNA sequence to be reverse complemented.

    Returns:
        str: The reverse complement of the input sequence.

    Example:
        seq = 'ATGC'
        print(reverseComplement(seq))  # Outputs: 'GCAT'

    Note:
        The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G). It also supports N (any base) and - (gap).
    """
    comp = []
    for i in (get_revComp_caseless(j) for j in seq):
        comp.append(i)
    return "".join(comp[::-1])


def simplifySeq(seq):
    """
    Simplifies a DNA sequence by replacing all nucleotides and IUPAC ambiguity codes with asterisks.

    This function takes a DNA sequence and returns a simplified version where all nucleotides (A, C, G, T) and IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V) are replaced with asterisks (*). The function is case-insensitive.

    Args:
        seq (str): The DNA sequence to be simplified.

    Returns:
        str: The simplified sequence, where all nucleotides and IUPAC ambiguity codes are replaced with asterisks (*).

    Example:
        seq = 'ATGCRYSWKMBDHVN'
        print(simplifySeq(seq))  # Outputs: '*************'

    Note:
        The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G).
    """
    temp = re.sub("[ACGT]", "", (seq).upper())
    return temp.translate(str.maketrans("RYSWKMBDHV", "**********"))


def seqCounter(seq):
    """
    Returns a dictionary of character counts in a DNA sequence.

    This function takes a DNA sequence and returns a dictionary where the keys are nucleotide characters and the values are their counts in the sequence. It also handles IUPAC ambiguity codes. The function is case-sensitive.

    Args:
        seq (str): The DNA sequence to be counted.

    Returns:
        dict: A dictionary where the keys are nucleotide characters and the values are their counts in the sequence. The dictionary also includes a 'VAR' key, which is the sum of the counts of all IUPAC ambiguity codes.

    Example:
        seq = 'ATGCRYSWKMBDHVN'
        print(seqCounter(seq))  # Outputs: {'A': 1, 'N': 1, '-': 0, 'C': 1, 'G': 1, 'T': 1, 'R': 1, 'Y': 1, 'S': 1, 'W': 1, 'K': 1, 'M': 1, 'B': 1, 'D': 1, 'H': 1, 'V': 1, 'VAR': 10}

    Note:
        The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G). It also supports N (any base) and - (gap).
    """
    d = {}
    d = {
        "A": 0,
        "N": 0,
        "-": 0,
        "C": 0,
        "G": 0,
        "T": 0,
        "R": 0,
        "Y": 0,
        "S": 0,
        "W": 0,
        "K": 0,
        "M": 0,
        "B": 0,
        "D": 0,
        "H": 0,
        "V": 0,
    }
    for c in seq:
        if c in d:
            d[c] += 1
    d["VAR"] = (
        d["R"]
        + d["Y"]
        + d["S"]
        + d["W"]
        + d["K"]
        + d["M"]
        + d["B"]
        + d["D"]
        + d["H"]
        + d["V"]
    )
    return d


def getFlankCounts(ref, x, y, dist):
    """
    Returns the counts of variants, gaps, and 'N's in the flanking regions of a substring within a reference sequence.

    This function takes a reference sequence and the start and end indices of a substring within the reference. It then counts the number of variants, gaps, and 'N's in the regions of the reference that flank the substring. The size of these flanking regions is determined by the 'dist' parameter.

    Args:
        ref (str): The reference sequence.

        x (int): The start index of the substring within the reference.

        y (int): The end index of the substring within the reference.

        dist (int): The distance from the substring to include in the flanking regions.

    Returns:
        dict: A dictionary with keys 'VAR', 'GAP', and 'N', and values corresponding to the counts of variants, gaps, and 'N's in the flanking regions, respectively.

    Example:
        ref = 'ATGCNNNATGC'
        x = 4
        y = 7
        dist = 2
        print(getFlankCounts(ref, x, y, dist))  # Outputs: {'VAR': 0, 'GAP': 0, 'N': 2}

    Note:
        The function assumes that variants are represented by '*', gaps by '-', and 'N's by 'N' in the reference sequence.
    """
    x2 = x - dist
    if x2 < 0:
        x2 = 0
    y2 = y + dist
    if y2 > len(ref):
        y2 = len(ref)
    flanks = ref[x2:x] + ref[y:y2]  # flanks = right + left flank
    counts = seqCounterSimple(simplifySeq(flanks))
    return counts


def seqCounterSimple(seq):
    """
    Returns the counts of variants, gaps, and 'N's in the flanking regions of a substring within a reference sequence.

    This function takes a reference sequence and the start and end indices of a substring within the reference. It then counts the number of variants, gaps, and 'N's in the regions of the reference that flank the substring. The size of these flanking regions is determined by the 'dist' parameter.

    Args:
        ref (str): The reference sequence.

        x (int): The start index of the substring within the reference.

        y (int): The end index of the substring within the reference.

        dist (int): The distance from the substring to include in the flanking regions.

    Returns:
        dict: A dictionary with keys 'VAR', 'GAP', and 'N', and values corresponding to the counts of variants, gaps, and 'N's in the flanking regions, respectively.

    Example:
        ref = 'ATGCNNNATGC'
        x = 4
        y = 7
        dist = 2
        print(getFlankCounts(ref, x, y, dist))  # Outputs: {'VAR': 0, 'GAP': 0, 'N': 2}

    Note:
        The function assumes that variants are represented by '*', gaps by '-', and 'N's by 'N' in the reference sequence.
    """
    d = {}
    d = {"N": 0, "-": 0, "*": 0}
    for c in seq:
        if c in d:
            d[c] += 1
    return d


def gc_counts(string):
    """Get GC content of a provided sequence."""
    new = re.sub("[GCgc]", "#", string)
    return sum(1 for c in new if c == "#")


def mask_counts(string):
    """Get counts of masked bases."""
    return sum(1 for c in string if c.islower())


def gc_content(string):
    """Get GC content as proportion."""
    new = re.sub("[GCgc]", "#", string)
    count = sum(1 for c in new if c == "#")
    return count / (len(string))


def mask_content(string):
    """Count number of lower case in a string."""
    count = sum(1 for c in string if c.islower())
    return count / (len(string))


def seqSlidingWindowString(seq, shift, width):
    """Generator to create sliding windows by slicing out substrings."""
    seqlen = len(seq)
    for i in range(0, seqlen, shift):
        if i + width > seqlen:
            j = seqlen
        else:
            j = i + width
        yield seq[i:j]
        if j == seqlen:
            break


def seqSlidingWindow(seq, shift, width):
    """Generator to create sliding windows by slicing out substrings."""
    seqlen = len(seq)
    for i in range(0, seqlen, shift):
        if i + width > seqlen:
            j = seqlen
        else:
            j = i + width
        yield [seq[i:j], i, j]
        if j == seqlen:
            break


def stringSubstitute(s, pos, c):
    """Fast way to replace single char in string.

    This way is a lot faster than doing it by making a list and subst in list.
    """
    return s[:pos] + c + s[pos + 1 :]


def listToSortUniqueString(l):
    """Get sorted unique string from list of chars.

    Args:
        l (List[str]): List of characters.

    Returns:
        List[str]: Sorted unique strings from list.
    """
    sl = sorted(set(l))
    return str("".join(sl))


def n_lower_chars(string):
    """Count number of lower case in a string."""
    return sum(1 for c in string if c.islower())


class slidingWindowGenerator:
    """
    An iterable object for creating a sliding window over a sequence.

    This class is used to create a sliding window over a sequence. The window moves over the sequence with a specified shift and width. The class is iterable, meaning it can be used in a for loop to iterate over all windows in the sequence.

    Attributes:
        _seq (str): The sequence over which the window is sliding.
        _seqlen (int): The length of the sequence.
        _shift (int): The number of positions to shift the window at each step.
        _width (int): The width of the window.
        _i (int): The current position of the window in the sequence.

    Example:
        >>>seq = 'ATGCATGC'
        >>>window = slidingWindowGenerator(seq, shift=1, width=3)
        >>>for w in window():
        >>>    print(w)  # Outputs: ['ATG', 0, 3], ['TGC', 1, 4], ['GCA', 2, 5], ...

    Note:
        The class is designed to be used with DNA sequences, but it can be used with any type of sequence.
    """

    # Need to come back and comment better...
    def __init__(self, seq, shift, width):
        self._seq = seq
        self._seqlen = len(self.__seq)
        self._shift = shift
        self._width = width
        self._i = 0

    def __call__(self):
        self._seqlen
        while self._i < self._seqlen:
            # print("i is ", self.__i, " : Base is ", self.__seq[self.__i]) #debug print
            if self._i + self._width > self._seqlen:
                j = self._seqlen
            else:
                j = self._i + self._width
            yield [self._seq[self._i : j], self._i, j]
            if j == self._seqlen:
                break
