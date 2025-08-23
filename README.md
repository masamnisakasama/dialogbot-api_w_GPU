## 現在の状況
ローカルでテストが通って、curlからもフロントからもAPIが叩けたので本番環境に移しています。<br>
しかしながら、MacではGPUは使えず、GPUを用いたバージョンはデプロイしてから実際にカールで叩くことでしか確かめられません。<br>
現在Cloud runをデプロイして機能をカールで確かめている段階です。ヘルスチェックには200が返ってきますが、STT系のAPIを叩くと500が返ってくるのでそこでかなり詰まっています。原因を切り分け中です。


<img width="721" height="271" alt="Screenshot 2025-08-23 at 12 43 59" src="https://github.com/user-attachments/assets/bbc0bd4c-fe19-49ec-a05f-2a74c2c4f385" /><br>
上がヘルスチェックで200、下がカールでFast APIのAPIに音声を投げて500になっている部分です。
今回GPUと相性の良いfaster-whisperを導入したのですが、それも原因かもしれません。
すなわち、faster-whisperはHugging Face+CTranslate2(CT2)+量子化設定(int8_float16)+CT2 の初回キャッシュ書き込みのような設定することが多い一方、Whisperは PyTorch+CUDA+ffmpegだけで動くのが原因かもしれません。焼き込み時にHugging Faceを既にダウンロードするなど試したが依然動かず厳しいので、whisper+GPUの方針でいくつもりです。
