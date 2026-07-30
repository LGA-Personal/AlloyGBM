# Multiclass Parallelism Benchmark

## Environment

- source commit: d3cb4a7922fcd650e1f0734433283319fc022fbe
- python: 3.13.5
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- logical cpus: 10

## Acceptance Contract

- Serial and parallel arms must have identical artifact and prediction hashes.
- Both arms must complete the same requested rounds with finite probabilities.
- Multiclass log loss must beat the class-prior baseline.
- Explicit worker requests are upper bounds; quick-run timing is descriptive.
- Full-run high-class median speedup must exceed 1.0x and low-class median regression must stay within 10%.

## Timing Summary

| Shape | Classes | Scenarios | Median speedup | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium-wide | 3 | 6 | 2.313x | 2.166x | 2.413x |
| medium-wide | 12 | 6 | 2.134x | 1.650x | 2.265x |
| small-control | 3 | 6 | 1.009x | 0.976x | 1.054x |
| small-control | 12 | 6 | 1.490x | 1.476x | 1.525x |
| tall-narrow | 3 | 6 | 1.322x | 1.307x | 1.369x |
| tall-narrow | 12 | 6 | 1.262x | 1.254x | 1.273x |

## Records

| Scenario | Arm | Workers | Eligible | Fit s | Rounds | Log loss | Prior loss | Artifact SHA-256 | Prediction SHA-256 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| tall-narrow-32768x8-3class-level-seed0 | serial | 1 | False | 0.097407 | 12 | 0.578623 | 1.094455 | `1af7fbdf4b20b8de209d356c2eb737fdd3c4853ecb93b4bc5b07d9218f0c1d9e` | `2b827e9dd4c0d1e69cc99deebdd83001dbb1cd539f25a3aa28c10b81b92dc2b4` |
| tall-narrow-32768x8-3class-level-seed0 | parallel | 4 | True | 0.073660 | 12 | 0.578623 | 1.094455 | `1af7fbdf4b20b8de209d356c2eb737fdd3c4853ecb93b4bc5b07d9218f0c1d9e` | `2b827e9dd4c0d1e69cc99deebdd83001dbb1cd539f25a3aa28c10b81b92dc2b4` |
| tall-narrow-32768x8-3class-level-seed1 | parallel | 4 | True | 0.074575 | 12 | 0.530681 | 1.093890 | `dbcc3f1fd16cb8e7e14afed9b3cf5bc9487f5eb713c425b2dabb8640ddd00c00` | `051ccf5a11c00e2703cb366deace411303d31750a1e2df3a663207b3c37c5808` |
| tall-narrow-32768x8-3class-level-seed1 | serial | 1 | False | 0.098534 | 12 | 0.530681 | 1.093890 | `dbcc3f1fd16cb8e7e14afed9b3cf5bc9487f5eb713c425b2dabb8640ddd00c00` | `051ccf5a11c00e2703cb366deace411303d31750a1e2df3a663207b3c37c5808` |
| tall-narrow-32768x8-3class-level-seed2 | serial | 1 | False | 0.098756 | 12 | 0.606401 | 1.088032 | `ac485efd969cb724f5e4a3fedb775a33b3d35afdf2160088824384c612197c93` | `8ff5a90ba5aa83c114d55da8313b30584b75eb854fdecc368f7de579841a5bf4` |
| tall-narrow-32768x8-3class-level-seed2 | parallel | 4 | True | 0.074081 | 12 | 0.606401 | 1.088032 | `ac485efd969cb724f5e4a3fedb775a33b3d35afdf2160088824384c612197c93` | `8ff5a90ba5aa83c114d55da8313b30584b75eb854fdecc368f7de579841a5bf4` |
| tall-narrow-32768x8-3class-leaf-seed0 | serial | 1 | False | 0.101732 | 12 | 0.578623 | 1.094455 | `6172e6191f73469758c05d33e423f2af2560eeef4ca6f96ccf89e32b416d9c95` | `2b827e9dd4c0d1e69cc99deebdd83001dbb1cd539f25a3aa28c10b81b92dc2b4` |
| tall-narrow-32768x8-3class-leaf-seed0 | parallel | 4 | True | 0.074296 | 12 | 0.578623 | 1.094455 | `6172e6191f73469758c05d33e423f2af2560eeef4ca6f96ccf89e32b416d9c95` | `2b827e9dd4c0d1e69cc99deebdd83001dbb1cd539f25a3aa28c10b81b92dc2b4` |
| tall-narrow-32768x8-3class-leaf-seed1 | parallel | 4 | True | 0.075205 | 12 | 0.530681 | 1.093890 | `cc5affe73174c770f35b61dedfe5bc4d41185de71d48ae9a26d30fed823e1b0d` | `051ccf5a11c00e2703cb366deace411303d31750a1e2df3a663207b3c37c5808` |
| tall-narrow-32768x8-3class-leaf-seed1 | serial | 1 | False | 0.098310 | 12 | 0.530681 | 1.093890 | `cc5affe73174c770f35b61dedfe5bc4d41185de71d48ae9a26d30fed823e1b0d` | `051ccf5a11c00e2703cb366deace411303d31750a1e2df3a663207b3c37c5808` |
| tall-narrow-32768x8-3class-leaf-seed2 | serial | 1 | False | 0.098669 | 12 | 0.606401 | 1.088032 | `7e5bd10b0cc32be79e9300e1c9ffb11bf62e3c32f174f59675d0fc4805972f22` | `8ff5a90ba5aa83c114d55da8313b30584b75eb854fdecc368f7de579841a5bf4` |
| tall-narrow-32768x8-3class-leaf-seed2 | parallel | 4 | True | 0.075434 | 12 | 0.606401 | 1.088032 | `7e5bd10b0cc32be79e9300e1c9ffb11bf62e3c32f174f59675d0fc4805972f22` | `8ff5a90ba5aa83c114d55da8313b30584b75eb854fdecc368f7de579841a5bf4` |
| tall-narrow-32768x8-12class-level-seed0 | serial | 1 | False | 0.438738 | 12 | 1.304768 | 2.465679 | `4fdf6a18fafd502752a24bddcab67552dabfc0c804ba815df345de7038423130` | `d28f10b98f3ff2bb49adb027e79154af9be9ecbb56989f1b3c55fc3ca1fb9a6f` |
| tall-narrow-32768x8-12class-level-seed0 | parallel | 4 | True | 0.349847 | 12 | 1.304768 | 2.465679 | `4fdf6a18fafd502752a24bddcab67552dabfc0c804ba815df345de7038423130` | `d28f10b98f3ff2bb49adb027e79154af9be9ecbb56989f1b3c55fc3ca1fb9a6f` |
| tall-narrow-32768x8-12class-level-seed1 | parallel | 4 | True | 0.344776 | 12 | 1.270480 | 2.452761 | `bf2f4a3bc63a727b9d9e32db474d66f3ddd1e60d55c0db4f4aa302a740705f7c` | `ad79ac5459b44c147a71f0636a1636f58cb4589b52ceb64ffe142b6cd97dd90a` |
| tall-narrow-32768x8-12class-level-seed1 | serial | 1 | False | 0.432780 | 12 | 1.270480 | 2.452761 | `bf2f4a3bc63a727b9d9e32db474d66f3ddd1e60d55c0db4f4aa302a740705f7c` | `ad79ac5459b44c147a71f0636a1636f58cb4589b52ceb64ffe142b6cd97dd90a` |
| tall-narrow-32768x8-12class-level-seed2 | serial | 1 | False | 0.441034 | 12 | 1.353678 | 2.459191 | `64d8a768d5bebbc41fbb198e9df0009ce4a8e1b06e9107d8f445cb9607ce2839` | `4dc6e4950b56898409aa72e7d6cee2709220217a43468cb0a914c4144a25b256` |
| tall-narrow-32768x8-12class-level-seed2 | parallel | 4 | True | 0.348631 | 12 | 1.353678 | 2.459191 | `64d8a768d5bebbc41fbb198e9df0009ce4a8e1b06e9107d8f445cb9607ce2839` | `4dc6e4950b56898409aa72e7d6cee2709220217a43468cb0a914c4144a25b256` |
| tall-narrow-32768x8-12class-leaf-seed0 | serial | 1 | False | 0.441090 | 12 | 1.304768 | 2.465679 | `a7083155dcf911983eda401160fef99cf1c77a6fbfb1699b1aaafafe706e0307` | `d28f10b98f3ff2bb49adb027e79154af9be9ecbb56989f1b3c55fc3ca1fb9a6f` |
| tall-narrow-32768x8-12class-leaf-seed0 | parallel | 4 | True | 0.350368 | 12 | 1.304768 | 2.465679 | `a7083155dcf911983eda401160fef99cf1c77a6fbfb1699b1aaafafe706e0307` | `d28f10b98f3ff2bb49adb027e79154af9be9ecbb56989f1b3c55fc3ca1fb9a6f` |
| tall-narrow-32768x8-12class-leaf-seed1 | parallel | 4 | True | 0.342061 | 12 | 1.270480 | 2.452761 | `7ea419bad6b12b04b79ad380b1d9eafab6ce3afa18fcfb2b27ef1571d91c4184` | `ad79ac5459b44c147a71f0636a1636f58cb4589b52ceb64ffe142b6cd97dd90a` |
| tall-narrow-32768x8-12class-leaf-seed1 | serial | 1 | False | 0.432529 | 12 | 1.270480 | 2.452761 | `7ea419bad6b12b04b79ad380b1d9eafab6ce3afa18fcfb2b27ef1571d91c4184` | `ad79ac5459b44c147a71f0636a1636f58cb4589b52ceb64ffe142b6cd97dd90a` |
| tall-narrow-32768x8-12class-leaf-seed2 | serial | 1 | False | 0.445674 | 12 | 1.353678 | 2.459191 | `eb35c1500175c8b531dbe1472cb9bf0068b2a383115a040d9d0228acb9f3260d` | `4dc6e4950b56898409aa72e7d6cee2709220217a43468cb0a914c4144a25b256` |
| tall-narrow-32768x8-12class-leaf-seed2 | parallel | 4 | True | 0.350187 | 12 | 1.353678 | 2.459191 | `eb35c1500175c8b531dbe1472cb9bf0068b2a383115a040d9d0228acb9f3260d` | `4dc6e4950b56898409aa72e7d6cee2709220217a43468cb0a914c4144a25b256` |
| medium-wide-4096x128-3class-level-seed0 | serial | 1 | False | 0.054874 | 12 | 0.955558 | 1.097662 | `c1982355d7168462ab48cddc1117a52cf26785886469a5b61e1308f2069c2729` | `49afd809fd67624b4e6b1609bbda8be84106ed471a18b38296021d562cf03881` |
| medium-wide-4096x128-3class-level-seed0 | parallel | 4 | True | 0.025329 | 12 | 0.955558 | 1.097662 | `c1982355d7168462ab48cddc1117a52cf26785886469a5b61e1308f2069c2729` | `49afd809fd67624b4e6b1609bbda8be84106ed471a18b38296021d562cf03881` |
| medium-wide-4096x128-3class-level-seed1 | parallel | 4 | True | 0.024300 | 12 | 0.961698 | 1.098450 | `82f89e96807ee0b3ea65518ffa77031d20a108c1b4d4922a440ad1f7ced2e0c1` | `7fc0e078afa2ac3c7d6994063b9a3b492cf2928a59852a40b63ce002b64e935f` |
| medium-wide-4096x128-3class-level-seed1 | serial | 1 | False | 0.058151 | 12 | 0.961698 | 1.098450 | `82f89e96807ee0b3ea65518ffa77031d20a108c1b4d4922a440ad1f7ced2e0c1` | `7fc0e078afa2ac3c7d6994063b9a3b492cf2928a59852a40b63ce002b64e935f` |
| medium-wide-4096x128-3class-level-seed2 | serial | 1 | False | 0.056581 | 12 | 0.968728 | 1.097620 | `488b6bf583419811ce06b247dbef2c41da1a71620752e7e3168e5b188ad71856` | `f023ce0683600aa30216304f427f997918a6800a04ef1d160b991d2cd6364f22` |
| medium-wide-4096x128-3class-level-seed2 | parallel | 4 | True | 0.024969 | 12 | 0.968728 | 1.097620 | `488b6bf583419811ce06b247dbef2c41da1a71620752e7e3168e5b188ad71856` | `f023ce0683600aa30216304f427f997918a6800a04ef1d160b991d2cd6364f22` |
| medium-wide-4096x128-3class-leaf-seed0 | serial | 1 | False | 0.059024 | 12 | 0.955558 | 1.097662 | `1eb91f9755d35b23eeb8cebcf52c2b0ff9702d19555936608f61dfede71a69af` | `49afd809fd67624b4e6b1609bbda8be84106ed471a18b38296021d562cf03881` |
| medium-wide-4096x128-3class-leaf-seed0 | parallel | 4 | True | 0.025373 | 12 | 0.955558 | 1.097662 | `1eb91f9755d35b23eeb8cebcf52c2b0ff9702d19555936608f61dfede71a69af` | `49afd809fd67624b4e6b1609bbda8be84106ed471a18b38296021d562cf03881` |
| medium-wide-4096x128-3class-leaf-seed1 | parallel | 4 | True | 0.024105 | 12 | 0.961698 | 1.098450 | `cbbb01ea41a924d301b306feb25f20e8d82ec922d1b9ab99aaf65563540d46a5` | `7fc0e078afa2ac3c7d6994063b9a3b492cf2928a59852a40b63ce002b64e935f` |
| medium-wide-4096x128-3class-leaf-seed1 | serial | 1 | False | 0.058169 | 12 | 0.961698 | 1.098450 | `cbbb01ea41a924d301b306feb25f20e8d82ec922d1b9ab99aaf65563540d46a5` | `7fc0e078afa2ac3c7d6994063b9a3b492cf2928a59852a40b63ce002b64e935f` |
| medium-wide-4096x128-3class-leaf-seed2 | serial | 1 | False | 0.057568 | 12 | 0.968728 | 1.097620 | `a54845163e7085437602980ed14343dc8cc015b1a332daef4c2b8ced31616d5f` | `f023ce0683600aa30216304f427f997918a6800a04ef1d160b991d2cd6364f22` |
| medium-wide-4096x128-3class-leaf-seed2 | parallel | 4 | True | 0.025023 | 12 | 0.968728 | 1.097620 | `a54845163e7085437602980ed14343dc8cc015b1a332daef4c2b8ced31616d5f` | `f023ce0683600aa30216304f427f997918a6800a04ef1d160b991d2cd6364f22` |
| medium-wide-4096x128-12class-level-seed0 | serial | 1 | False | 0.206676 | 12 | 2.201348 | 2.481546 | `eea4af5b07669baa968c0511223cf4df9411da348d3eab37da4c5a3c346a3d51` | `3d453f86f6ef35188028e879a2538711c0b8a1cf785bf5d097d248ad7a9ee8dd` |
| medium-wide-4096x128-12class-level-seed0 | parallel | 4 | True | 0.092321 | 12 | 2.201348 | 2.481546 | `eea4af5b07669baa968c0511223cf4df9411da348d3eab37da4c5a3c346a3d51` | `3d453f86f6ef35188028e879a2538711c0b8a1cf785bf5d097d248ad7a9ee8dd` |
| medium-wide-4096x128-12class-level-seed1 | parallel | 4 | True | 0.092472 | 12 | 2.241070 | 2.482758 | `4541c3b6aef8610e712c0e22314ca0956c1219f0b1d144133a63052bd0d7edbb` | `24c1aa4d5f661004480e94df13399d3da4c4f397c34db8fb7e258fc9c5787f72` |
| medium-wide-4096x128-12class-level-seed1 | serial | 1 | False | 0.209466 | 12 | 2.241070 | 2.482758 | `4541c3b6aef8610e712c0e22314ca0956c1219f0b1d144133a63052bd0d7edbb` | `24c1aa4d5f661004480e94df13399d3da4c4f397c34db8fb7e258fc9c5787f72` |
| medium-wide-4096x128-12class-level-seed2 | serial | 1 | False | 0.201091 | 12 | 2.238170 | 2.490369 | `23b37b1a8172b8fa42e9b798c3db4b67b299eadf57afadeefddd70046bcb0bfb` | `6cdb4c2c85f3d0a84d10ebfe1aafc20eb98ff45cbad3da35b770930d7994dbbe` |
| medium-wide-4096x128-12class-level-seed2 | parallel | 4 | True | 0.093799 | 12 | 2.238170 | 2.490369 | `23b37b1a8172b8fa42e9b798c3db4b67b299eadf57afadeefddd70046bcb0bfb` | `6cdb4c2c85f3d0a84d10ebfe1aafc20eb98ff45cbad3da35b770930d7994dbbe` |
| medium-wide-4096x128-12class-leaf-seed0 | serial | 1 | False | 0.207106 | 12 | 2.201348 | 2.481546 | `c6304272c2ba174518fe739c1d53a94150d8fb7602bc4073728f05b948997ed1` | `3d453f86f6ef35188028e879a2538711c0b8a1cf785bf5d097d248ad7a9ee8dd` |
| medium-wide-4096x128-12class-leaf-seed0 | parallel | 4 | True | 0.097538 | 12 | 2.201348 | 2.481546 | `c6304272c2ba174518fe739c1d53a94150d8fb7602bc4073728f05b948997ed1` | `3d453f86f6ef35188028e879a2538711c0b8a1cf785bf5d097d248ad7a9ee8dd` |
| medium-wide-4096x128-12class-leaf-seed1 | parallel | 4 | True | 0.127174 | 12 | 2.241070 | 2.482758 | `62cef4326e5af0a40c92c710e843492f9109365a61ace8e4beeb327197ac6325` | `24c1aa4d5f661004480e94df13399d3da4c4f397c34db8fb7e258fc9c5787f72` |
| medium-wide-4096x128-12class-leaf-seed1 | serial | 1 | False | 0.209895 | 12 | 2.241070 | 2.482758 | `62cef4326e5af0a40c92c710e843492f9109365a61ace8e4beeb327197ac6325` | `24c1aa4d5f661004480e94df13399d3da4c4f397c34db8fb7e258fc9c5787f72` |
| medium-wide-4096x128-12class-leaf-seed2 | serial | 1 | False | 0.202505 | 12 | 2.238170 | 2.490369 | `e1d9e7781d0fc34418fc8db851d18f514793a91eeb57fc26b0d88e63c542e6c4` | `6cdb4c2c85f3d0a84d10ebfe1aafc20eb98ff45cbad3da35b770930d7994dbbe` |
| medium-wide-4096x128-12class-leaf-seed2 | parallel | 4 | True | 0.095298 | 12 | 2.238170 | 2.490369 | `e1d9e7781d0fc34418fc8db851d18f514793a91eeb57fc26b0d88e63c542e6c4` | `6cdb4c2c85f3d0a84d10ebfe1aafc20eb98ff45cbad3da35b770930d7994dbbe` |
| small-control-512x8-3class-level-seed0 | serial | 1 | False | 0.002695 | 12 | 0.610389 | 1.097037 | `2fa170d476e04a906e3eba6214b72def915662773a2f2f28784c5d914542faa4` | `993e9aad210f1416d766c2b48da21b8a26fba399fa829e797726e198c6e7ff32` |
| small-control-512x8-3class-level-seed0 | parallel | 4 | False | 0.002762 | 12 | 0.610389 | 1.097037 | `2fa170d476e04a906e3eba6214b72def915662773a2f2f28784c5d914542faa4` | `993e9aad210f1416d766c2b48da21b8a26fba399fa829e797726e198c6e7ff32` |
| small-control-512x8-3class-level-seed1 | parallel | 4 | False | 0.002593 | 12 | 0.537805 | 1.095533 | `1d691aa0f2b1e4848b73fcd070b70e057338db30dc7ecb06f32508f4ce874b20` | `4d37ecc6d585854f4e7fccb208d7058a83a9c7a2419d6d09cc15634ac3791e23` |
| small-control-512x8-3class-level-seed1 | serial | 1 | False | 0.002636 | 12 | 0.537805 | 1.095533 | `1d691aa0f2b1e4848b73fcd070b70e057338db30dc7ecb06f32508f4ce874b20` | `4d37ecc6d585854f4e7fccb208d7058a83a9c7a2419d6d09cc15634ac3791e23` |
| small-control-512x8-3class-level-seed2 | serial | 1 | False | 0.002657 | 12 | 0.577733 | 1.084821 | `a78fe13790bb850ab1d5dc98760789a2fe9636a8cab2dd153ede3c6f644196c3` | `d53a3878937164041cbe7dcb18b6aeba230f6641467ea27947844b3ab24cbd45` |
| small-control-512x8-3class-level-seed2 | parallel | 4 | False | 0.002521 | 12 | 0.577733 | 1.084821 | `a78fe13790bb850ab1d5dc98760789a2fe9636a8cab2dd153ede3c6f644196c3` | `d53a3878937164041cbe7dcb18b6aeba230f6641467ea27947844b3ab24cbd45` |
| small-control-512x8-3class-leaf-seed0 | serial | 1 | False | 0.002590 | 12 | 0.610389 | 1.097037 | `d494ad4034485d0129938bfc8ef43bfe8ef1031c69ddf9d17f37c57665676dd5` | `993e9aad210f1416d766c2b48da21b8a26fba399fa829e797726e198c6e7ff32` |
| small-control-512x8-3class-leaf-seed0 | parallel | 4 | False | 0.002589 | 12 | 0.610389 | 1.097037 | `d494ad4034485d0129938bfc8ef43bfe8ef1031c69ddf9d17f37c57665676dd5` | `993e9aad210f1416d766c2b48da21b8a26fba399fa829e797726e198c6e7ff32` |
| small-control-512x8-3class-leaf-seed1 | parallel | 4 | False | 0.002506 | 12 | 0.537805 | 1.095533 | `656f46bd43acf159c20bb736e0ec0d621f4b9ef415f188571efd833878a0cc2b` | `4d37ecc6d585854f4e7fccb208d7058a83a9c7a2419d6d09cc15634ac3791e23` |
| small-control-512x8-3class-leaf-seed1 | serial | 1 | False | 0.002523 | 12 | 0.537805 | 1.095533 | `656f46bd43acf159c20bb736e0ec0d621f4b9ef415f188571efd833878a0cc2b` | `4d37ecc6d585854f4e7fccb208d7058a83a9c7a2419d6d09cc15634ac3791e23` |
| small-control-512x8-3class-leaf-seed2 | serial | 1 | False | 0.002536 | 12 | 0.577733 | 1.084821 | `fe6cc3a604cb08de5d7938038e2cbe084b08df2c75fd21d6400589a7067dbe70` | `d53a3878937164041cbe7dcb18b6aeba230f6641467ea27947844b3ab24cbd45` |
| small-control-512x8-3class-leaf-seed2 | parallel | 4 | False | 0.002507 | 12 | 0.577733 | 1.084821 | `fe6cc3a604cb08de5d7938038e2cbe084b08df2c75fd21d6400589a7067dbe70` | `d53a3878937164041cbe7dcb18b6aeba230f6641467ea27947844b3ab24cbd45` |
| small-control-512x8-12class-level-seed0 | serial | 1 | False | 0.010657 | 12 | 1.407656 | 2.472664 | `a1450ffcffbc8501b0c63a74e3bbce5cf3e23c46a63b02f61a8ae036ca280531` | `ed59d361b466fdae21a3cf3d7b5538b546a1d4aef3236d31e8170ced75c055e8` |
| small-control-512x8-12class-level-seed0 | parallel | 4 | True | 0.006990 | 12 | 1.407656 | 2.472664 | `a1450ffcffbc8501b0c63a74e3bbce5cf3e23c46a63b02f61a8ae036ca280531` | `ed59d361b466fdae21a3cf3d7b5538b546a1d4aef3236d31e8170ced75c055e8` |
| small-control-512x8-12class-level-seed1 | parallel | 4 | True | 0.006994 | 12 | 1.411820 | 2.452256 | `f165e716b3c6fe6be6545c87ab8f77fd84f622d747ea259152af48a41cf61960` | `fa16a3194c6af724c52e7f992eb64ae013aa5feca6b74c95515c3cb1c4d62f48` |
| small-control-512x8-12class-level-seed1 | serial | 1 | False | 0.010521 | 12 | 1.411820 | 2.452256 | `f165e716b3c6fe6be6545c87ab8f77fd84f622d747ea259152af48a41cf61960` | `fa16a3194c6af724c52e7f992eb64ae013aa5feca6b74c95515c3cb1c4d62f48` |
| small-control-512x8-12class-level-seed2 | serial | 1 | False | 0.010541 | 12 | 1.507663 | 2.455156 | `74b2e4a420d9c7bfc63e6f48f40b52278a8ddee6a64c1b0dfe2aaf7904eb84fc` | `f781048aafd158c0bdb8a8a481e6281e93af4b7c129bffd56c9ee4e30987e9ef` |
| small-control-512x8-12class-level-seed2 | parallel | 4 | True | 0.007097 | 12 | 1.507663 | 2.455156 | `74b2e4a420d9c7bfc63e6f48f40b52278a8ddee6a64c1b0dfe2aaf7904eb84fc` | `f781048aafd158c0bdb8a8a481e6281e93af4b7c129bffd56c9ee4e30987e9ef` |
| small-control-512x8-12class-leaf-seed0 | serial | 1 | False | 0.010478 | 12 | 1.407656 | 2.472664 | `0d67f7557a864cf086d2daa364716d7eeee99107996adaa21ea29fc149f2b2b9` | `ed59d361b466fdae21a3cf3d7b5538b546a1d4aef3236d31e8170ced75c055e8` |
| small-control-512x8-12class-leaf-seed0 | parallel | 4 | True | 0.007012 | 12 | 1.407656 | 2.472664 | `0d67f7557a864cf086d2daa364716d7eeee99107996adaa21ea29fc149f2b2b9` | `ed59d361b466fdae21a3cf3d7b5538b546a1d4aef3236d31e8170ced75c055e8` |
| small-control-512x8-12class-leaf-seed1 | parallel | 4 | True | 0.006897 | 12 | 1.411820 | 2.452256 | `0810e4699518771753e0318823a23167eff3cc5370622a6e6eac6d55175b95dd` | `fa16a3194c6af724c52e7f992eb64ae013aa5feca6b74c95515c3cb1c4d62f48` |
| small-control-512x8-12class-leaf-seed1 | serial | 1 | False | 0.010206 | 12 | 1.411820 | 2.452256 | `0810e4699518771753e0318823a23167eff3cc5370622a6e6eac6d55175b95dd` | `fa16a3194c6af724c52e7f992eb64ae013aa5feca6b74c95515c3cb1c4d62f48` |
| small-control-512x8-12class-leaf-seed2 | serial | 1 | False | 0.010191 | 12 | 1.507663 | 2.455156 | `32bd355b8e00cbeb6bd54f4d6140587ae4dfdbe9b18d856cf795f8f8720aadbc` | `f781048aafd158c0bdb8a8a481e6281e93af4b7c129bffd56c9ee4e30987e9ef` |
| small-control-512x8-12class-leaf-seed2 | parallel | 4 | True | 0.006903 | 12 | 1.507663 | 2.455156 | `32bd355b8e00cbeb6bd54f4d6140587ae4dfdbe9b18d856cf795f8f8720aadbc` | `f781048aafd158c0bdb8a8a481e6281e93af4b7c129bffd56c9ee4e30987e9ef` |

## Gate

- Failures: 0
