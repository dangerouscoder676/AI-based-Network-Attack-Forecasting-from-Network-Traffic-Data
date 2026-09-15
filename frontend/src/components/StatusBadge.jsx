export default function StatusBadge({ children, tone = "safe" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
