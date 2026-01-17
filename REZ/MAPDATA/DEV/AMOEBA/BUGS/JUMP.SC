Name "Jump"

AbortOnDamage		FALSE	//	abort the script if we're damaged
AbortOnEnemySighted FALSE	//	Don't abort if we sight an enemy
TimeOut 999.0
FaceMoveDir			TRUE

Loop1:

StartAnim "AttackB"

Jump	420

Wait	3.0

Goto	Loop1

End