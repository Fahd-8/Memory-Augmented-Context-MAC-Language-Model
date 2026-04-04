interface Props {
    onReset: () => void
    loading: boolean
  }
  
  export default function ResetButton({ onReset, loading }: Props) {
    return (
      <button
        onClick={onReset}
        disabled={loading}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 12px',
          background: 'transparent',
          border: '1px solid var(--border)',
          borderRadius: 6,
          color: 'var(--text-3)',
          fontFamily: 'IBM Plex Mono',
          fontSize: 11,
          letterSpacing: '0.04em',
          cursor: loading ? 'not-allowed' : 'pointer',
          opacity: loading ? 0.4 : 1,
          transition: 'all 0.15s ease',
        }}
        onMouseEnter={e => {
          if (!loading) {
            (e.target as HTMLElement).closest('button')!.style.borderColor = 'var(--border-light)'
            ;(e.target as HTMLElement).closest('button')!.style.color = 'var(--text-2)'
          }
        }}
        onMouseLeave={e => {
          (e.target as HTMLElement).closest('button')!.style.borderColor = 'var(--border)'
          ;(e.target as HTMLElement).closest('button')!.style.color = 'var(--text-3)'
        }}
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
          <path d="M3 3v5h5"/>
        </svg>
        RESET MEMORY
      </button>
    )
  }