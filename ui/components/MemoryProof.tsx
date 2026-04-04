import { CompareResult } from '@/types'

interface Props {
  result: CompareResult | null
  loading: boolean
  onProve: () => void
}

export default function MemoryProof({ result, loading, onProve }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{
          fontFamily: 'IBM Plex Mono',
          fontSize: 12,
          color: 'var(--text-2)',
          letterSpacing: '0.1em',
          marginBottom: 6
        }}>MEMORY PROOF</div>
        <p style={{ color: 'var(--text-3)', fontSize: 12, lineHeight: 1.6 }}>
          Same question. TTT on vs off. If the LMM is working, only one will remember.
        </p>
      </div>

      <button
        onClick={onProve}
        disabled={loading}
        style={{
          width: '100%',
          padding: '10px 0',
          background: loading ? 'var(--bg-3)' : 'var(--bg-2)',
          border: '1px solid var(--border-light)',
          borderRadius: 6,
          color: loading ? 'var(--text-3)' : 'var(--text)',
          fontFamily: 'IBM Plex Mono',
          fontSize: 11,
          letterSpacing: '0.08em',
          cursor: loading ? 'not-allowed' : 'pointer',
          marginBottom: 20,
          transition: 'all 0.15s ease',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8
        }}
      >
        {loading ? '■ RUNNING TEST...' : '→ RUN MEMORY TEST'}
      </button>

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1, overflowY: 'auto' }}>
          <div style={{
            background: 'var(--green-dim)',
            border: '1px solid rgba(74, 222, 128, 0.2)',
            borderRadius: 6,
            padding: 14
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)' }}/>
              <span style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: 10,
                color: 'var(--green)',
                letterSpacing: '0.08em'
              }}>TTT=ON · LMM ACTIVE</span>
            </div>
            <p style={{
              fontFamily: 'IBM Plex Mono',
              fontSize: 12,
              color: 'var(--text-2)',
              lineHeight: 1.7,
              whiteSpace: 'pre-wrap'
            }}>{result.ttt_on}</p>
          </div>

          <div style={{
            background: 'var(--red-dim)',
            border: '1px solid rgba(248, 113, 113, 0.15)',
            borderRadius: 6,
            padding: 14
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)' }}/>
              <span style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: 10,
                color: 'var(--red)',
                letterSpacing: '0.08em'
              }}>TTT=OFF · NO MEMORY</span>
            </div>
            <p style={{
              fontFamily: 'IBM Plex Mono',
              fontSize: 12,
              color: 'var(--text-2)',
              lineHeight: 1.7,
              whiteSpace: 'pre-wrap'
            }}>{result.ttt_off}</p>
          </div>

          <div style={{
            padding: '8px 12px',
            background: 'var(--bg-2)',
            border: '1px solid var(--border)',
            borderRadius: 4
          }}>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--text-3)' }}>QUERY: </span>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--text-2)' }}>"{result.prompt}"</span>
          </div>
        </div>
      )}

      {!result && !loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontFamily: 'IBM Plex Mono',
              fontSize: 11,
              color: 'var(--text-3)',
              lineHeight: 1.8
            }}>
              <div>01001101 01000001 01000011</div>
              <div style={{ marginTop: 12, fontSize: 11 }}>
                build memory first<br/>then run the test
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}