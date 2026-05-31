import Lake
open Lake DSL

package "BvdProofs" where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require mdgen from git
  "https://github.com/Seasawher/mdgen" @ "main"

require "leanprover-community" / "mathlib"

@[default_target]
lean_lib «BvdProofs» where
  globs := #[.one `BvdProofs, .submodules `BvdProofs]
