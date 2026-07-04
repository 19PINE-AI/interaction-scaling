import { useEffect, useState } from 'react'

export const BASE = import.meta.env.BASE_URL

export const C = {
  gnd: '#2A78D6', gndDk: '#1B5A9E', ung: '#EB6834', ungDk: '#D1495B',
  stu: '#4A3AA7', ink: '#21201C', ink2: '#57534A', ink3: '#8A857A',
  grid: '#E9E5DC', good: '#2E7D46', bad: '#C03B2B', paper: '#FAF8F4', card: '#FFFFFF',
}

export function useSite() {
  const [site, setSite] = useState(null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    fetch(`${BASE}data/v2/site.json`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(setSite)
      .catch(() => setErr(true))
  }, [])
  return { site, err }
}

const caseCache = new Map()
export function useCase(mod, id) {
  const key = `${mod}/${id}`
  const [data, setData] = useState(caseCache.get(key) || null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    if (!mod || !id) return
    if (caseCache.has(key)) { setData(caseCache.get(key)); return }
    setData(null); setErr(false)
    fetch(`${BASE}data/v2/cases/${mod}/${id}.json`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => { caseCache.set(key, d); setData(d) })
      .catch(() => setErr(true))
  }, [key])
  return { data, err }
}

// tiny hash router: '' (home) or ['case', mod, id]
export function useRoute() {
  const parse = () => window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  const [route, setRoute] = useState(parse)
  useEffect(() => {
    const fn = () => setRoute(parse())
    window.addEventListener('hashchange', fn)
    return () => window.removeEventListener('hashchange', fn)
  }, [])
  return route
}

export const fmtTok = n => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}K` : `${n}`
