import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MAC — Memory-Augmented Context',
  description: 'Titans architecture. Neural long-term memory. Test-time training.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}