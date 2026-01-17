Name "strong"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
TimeOut				999.0
WalkSpeed			300
YawRate				360.0


StartAnim "AmbA"
Wait       3
StartAnim "AttackB"
Wait       1.3
StartAnim "AmbA"
Wait       0.5
StartAnim "AttackB"
Wait       1.3
StartAnim "AmbA"
Wait       1
StartAnim "AttackA"
Wait       2.4
StartAnim "AmbA"
Wait       3.2
StartAnim "AttackB"
Wait       1.3
StartAnim "Run"

MoveTo strongspot1

TurnTo strongspot2
MoveTo strongspot2

StartAnim "AmbA"

End



