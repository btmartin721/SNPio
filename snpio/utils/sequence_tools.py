import re
from collections import Counter
from typing import Any, Dict, List


def remove_items(all_list: List[Any], bad_list: List[Any]) -> List[Any]:
    """Removes items from a list based on another list.

    This function takes a list and removes any items that are present in a second list.

    Args:
        all_list (List[Any]): The list from which items are to be removed.

        bad_list (List[Any]): The list containing items to be removed from the first list.

    Returns:
        List[Any]: The first list with any items present in the second list removed.

    Example:
        >>> all_list = ['a', 'b', 'c', 'd']
        >>> bad_list = ['b', 'd']
        >>> print(remove_items(all_list, bad_list))
        >>> # Outputs: ['a', 'c']
    """  # using list comprehension to perform the task
    return [i for i in all_list if i not in bad_list]


def count_alleles(l, vcf: bool = False) -> int:
    """Counts the total number of unique alleles in a list of genotypes.

    This function takes a list of IUPAC or VCF-style (e.g. 0/1) genotypes and returns the total number of unique alleles. The genotypes can be in VCF or STRUCTURE-style format.

    Args:
        l (List[str]): A list of IUPAC or VCF-style genotypes.

        vcf (bool, optional): If True, the genotypes are in VCF format. If False, the genotypes are in STRUCTURE-style format. Defaults to False.

    Returns:
        int: The total number of unique alleles in the list.

    Example:
        >>> l = ['A/A', 'A/T', 'T/T', 'A/A', 'A/T']
        >>> print(count_alleles(l, vcf=True))
        >>> # Outputs: 2

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


def get_major_allele(
    l: List[str], num: int | None = None, vcf: bool = False
) -> List[str]:
    """
    Returns the most common alleles in a list.

    This function takes a list of genotypes for one sample and returns the most common alleles in descending order. The alleles can be in VCF or STRUCTURE-style format.

    Args:
        l (List[str]): A list of genotypes for one sample.

        num (int | None, optional): The number of elements to return. If None, all elements are returned. Defaults to None.

        vcf (bool, optional): If True, the alleles are in VCF format. If False, the alleles are in STRUCTURE-style format. Defaults to False.

    Returns:
        List[str]: The most common alleles in descending order.

    Example:
        >>> l = ['A/A', 'A/T', 'T/T', 'A/A', 'A/T']
        >>> print(get_major_allele(l, vcf=True))  # Outputs: ['A', 'T']

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


def get_iupac_caseless(char: str) -> List[str]:
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


def get_revComp_caseless(char: str) -> str:
    """
    Returns the reverse complement of a nucleotide, while preserving case.

    This function takes a nucleotide character and returns its reverse complement according to the standard DNA base pairing rules. It also handles IUPAC ambiguity codes. The case of the input character is preserved in the output.

    Args:
        char (str): The nucleotide character to be reverse complemented. Can be uppercase or lowercase.

    Returns:
        str: The reverse complement of the input character, with the same case.

    Example:
        >>> char = 'a'
        >>> print(get_revComp_caseless(char))
        >>> # Outputs: 't'

    Note:
        - The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G). It also supports N (any base) and - (gap).
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


def simplifySeq(seq: str) -> str:
    """Simplifies a DNA sequence by replacing all nucleotides and IUPAC ambiguity codes with asterisks.

    This function takes a DNA sequence and returns a simplified version where all nucleotides (A, C, G, T) and IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V) are replaced with asterisks (*). The function is case-insensitive.

    Args:
        seq (str): The DNA sequence to be simplified.

    Returns:
        str: The simplified sequence, where all nucleotides and IUPAC ambiguity codes are replaced with asterisks (*).

    Example:
        >>> seq = 'ATGCRYSWKMBDHVN'
        >>> print(simplifySeq(seq))
        >>> # Outputs: '*************'

    Note:
        The function supports the following IUPAC ambiguity codes: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C), B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G).
    """
    temp = re.sub("[ACGT]", "", (seq).upper())
    return temp.translate(str.maketrans("RYSWKMBDHV", "**********"))


def seqCounter(seq: str) -> Dict[str, int]:
    """
    Returns a dictionary of character counts in a DNA sequence.

    This function takes a DNA sequence and returns a dictionary where the keys are nucleotide characters and the values are their counts in the sequence. It also handles IUPAC ambiguity codes. The function is case-sensitive.

    Args:
        seq (str): The DNA sequence to be counted.

    Returns:
        Dict[str, int]: A dictionary where the keys are nucleotide characters and the values are their counts in the sequence. The dictionary also includes a 'VAR' key, which is the sum of the counts of all IUPAC ambiguity codes.

    Example:
        >>> seq = 'ATGCRYSWKMBDHVN'
        >>> print(seqCounter(seq))
        {'A': 1, 'N': 1, '-': 0, 'C': 1, 'G': 1, 'T': 1, 'R': 1, 'Y': 1, 'S': 1, 'W': 1, 'K': 1, 'M': 1, 'B': 1, 'D': 1, 'H': 1, 'V': 1, 'VAR': 10}

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
