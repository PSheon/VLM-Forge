// Word-level diff of two answers (LCS), for query / caption comparisons.

export default function TextDiff({ a, b }: { a: string; b: string }) {
  const A = a.split(/\s+/).filter(Boolean), B = b.split(/\s+/).filter(Boolean);
  const n = A.length, m = B.length;
  const L: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: React.ReactNode[] = [];
  let i = 0, j = 0, k = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push(<span key={k++}>{A[i]} </span>); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { out.push(<del key={k++}>{A[i]} </del>); i++; }
    else { out.push(<ins key={k++}>{B[j]} </ins>); j++; }
  }
  while (i < n) out.push(<del key={k++}>{A[i++]} </del>);
  while (j < m) out.push(<ins key={k++}>{B[j++]} </ins>);
  return <div className="diff">{out}</div>;
}
